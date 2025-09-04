import math
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from PIL import Image
from pathlib import Path
from argparse import ArgumentParser
from omegaconf import OmegaConf
from tqdm import tqdm
import time

from model import Unet
# from train_upsample import sample_ode  # 既存のサンプラーを利用

eps = 1e-3

@torch.no_grad()
def sample_ode(
    model,
    image_size,
    batch_size=16,
    channels=1,
    x_init=None,
    steps=1,          # ステップ数（増やすほど精度↑ ただし時間↑）
    t0=eps,
    t1=1.0,
    amp=True,          # autocastで高速&省メモリ
    known_mask=None,   # 既知画素(=1)を保持したいときのオプション
    x_known=None,      # 元のスパース入力（known_maskと同形）
):
    """
    dx/dt = v_theta(x, t) をオイラー法で t=t0→t1 に積分
    model: v = model(x, t) を返す U-Net
    返り値: x in [-1, 1], shape=(B,C,H,W)
    """
    device = next(model.parameters()).device
    shape = (batch_size, channels, image_size, image_size)
    b = shape[0]

    x = x_init.to(device) if x_init is not None else torch.randn(shape, device=device)
    model.eval()

    # 時間グリッド（線形）。必要ならコサイン等のスケジュールに置き換えてOK
    ts = torch.linspace(t0, t1, steps + 1, device=device)

    for i in range(steps):
        t = ts[i].expand(b)          # (B,)
        h = (ts[i+1] - ts[i]).item() # スカラー刻み幅

        if amp:
            with torch.cuda.amp.autocast():
                v = model(x, t)      # v_theta(x,t)
        else:
            v = model(x, t)

        x = x + h * v                # ★ オイラー更新

        # 既知画素の保持（スパース→デンスで堅牢にしたい場合）
        if known_mask is not None and x_known is not None:
            x = known_mask * x_known.to(device) + (1 - known_mask) * x

        # 断片化が気になるほど大きいときは適度にキャッシュを掃除（多用しない）
        # if (i + 1) % 10 == 0:
        #     torch.cuda.empty_cache()

    return x.clamp(-1, 1)

# ---- 追加: [-1,1] に正規化するだけ（サイズ変更なし） ----
class ToTensorMinus1To1:
    def __call__(self, img):
        t = T.ToTensor()(img)    # [0,1]
        return t * 2 - 1         # [-1,1]

# ---- 追加: 2Dハニング窓（エッジがゼロになり過ぎないよう下限を持たせる）----
def make_blend_window(h, w, device, floor=0.1):
    win_h = torch.hann_window(h, periodic=False, device=device)
    win_w = torch.hann_window(w, periodic=False, device=device)
    w2d = torch.outer(win_h, win_w).clamp_min(floor)  # (h,w)
    return w2d.view(1, 1, h, w)

# ---- 追加: タイル推論本体（sample_ode をタイルに対して繰り返し実行）----
@torch.no_grad()
def tiled_sample_ode(model, x_full, channels, tile=512, overlap=64, tile_batch=4):
    """
    x_full: (1,C,H,W) [-1,1]
    return: (1,C,H,W) [-1,1]
    """
    assert x_full.ndim == 4 and x_full.size(0) == 1
    device = x_full.device
    _, C, H, W = x_full.shape
    assert C == channels, f"channels mismatch: {C} vs {channels}"

    # 画像がタイルより小さい場合のみパディング（reflect）
    pad_top = max(0, (tile - H) // 2)
    pad_bottom = max(0, tile - H - pad_top)
    pad_left = max(0, (tile - W) // 2)
    pad_right = max(0, tile - W - pad_left)
    if pad_top or pad_bottom or pad_left or pad_right:
        x_pad = F.pad(x_full, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
    else:
        x_pad = x_full
    _, _, Hp, Wp = x_pad.shape

    stride = max(1, tile - overlap)
    # タイル開始位置（端は必ずカバーする）
    def positions(L):
        pos = list(range(0, max(L - tile, 0) + 1, stride))
        if len(pos) == 0 or pos[-1] != L - tile:
            pos.append(L - tile)
        return pos
    tops = positions(Hp)
    lefts = positions(Wp)

    weight = make_blend_window(tile, tile, device=device)  # (1,1,tile,tile)
    out_acc = torch.zeros((1, C, Hp, Wp), device=device)
    w_acc   = torch.zeros((1, 1, Hp, Wp), device=device)

    patches = []
    coords = []

    def flush():
        nonlocal patches, coords, out_acc, w_acc
        if not patches:
            return
        batch = torch.cat(patches, dim=0)  # (B,C,tile,tile)
        # 既存の sample_ode をそのまま利用
        y = sample_ode(model, image_size=tile, batch_size=batch.size(0),
                       channels=channels, x_init=batch)  # (B,C,tile,tile), [-1,1]
        for k, (top, left) in enumerate(coords):
            out_acc[:, :, top:top+tile, left:left+tile] += y[k:k+1] * weight
            w_acc[:, :, top:top+tile, left:left+tile] += weight
        patches = []
        coords = []

    # 走査
    for top in tops:
        for left in lefts:
            patch = x_pad[:, :, top:top+tile, left:left+tile]
            patches.append(patch)
            coords.append((top, left))
            if len(patches) >= tile_batch:
                flush()
    flush()

    out = out_acc / w_acc.clamp_min(1e-6)
    # パディング分をトリムして元サイズに戻す
    out = out[:, :, pad_top:pad_top+H, pad_left:pad_left+W]
    return out

# ---- 元のDataset（軽修正: 変換のみ差し替え） ----
class SparseImageDataset(torch.utils.data.Dataset):
    def __init__(self, sparse_dir, transform, img_convert="RGB"):
        self.sparse_dir = Path(sparse_dir)
        self.transform = transform
        self.img_convert = img_convert
        self.filenames = sorted([f.name for f in self.sparse_dir.glob("*")])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = Image.open(self.sparse_dir / fname).convert(self.img_convert)
        img = self.transform(img)
        return img, fname

def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--sparse_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ★ 変更: 画像は元解像度のまま [-1,1] 化（リサイズ/プーリングしない）
    transform = ToTensorMinus1To1()

    dataset = SparseImageDataset(args.sparse_dir, transform, getattr(config, "img_convert", "RGB"))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    model = Unet(**config.model).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ★ タイル設定（configに無ければデフォルトを使用）
    tile = int(getattr(config, "tile_size", 300))
    overlap = int(getattr(config, "tile_overlap", 24))
    tile_batch = int(getattr(config, "tile_batch", 4))
    channels = config.model.channels

    for x_sparse, fname in tqdm(loader, desc="Tiled Inference"):
        x_sparse = x_sparse.to(device)  # (1,C,H,W), [-1,1]
        start_time = time.time()
        with torch.no_grad():
            # ★ 変更: タイル推論でサンプリング
            result = tiled_sample_ode(
                model, x_sparse, channels=channels,
                tile=tile, overlap=overlap, tile_batch=tile_batch
            )
        elapsed = time.time() - start_time
        print(f"Image {fname[0]} generated in {elapsed:.2f} seconds")

        # 可視化保存（左: 入力, 右: 出力）
        inp_img = (x_sparse[0].clamp(-1, 1).cpu() + 1) / 2
        out_img = (result[0].clamp(-1, 1).cpu() + 1) / 2
        inp_pil = T.ToPILImage()(inp_img)
        out_pil = T.ToPILImage()(out_img)

        w, h = inp_pil.size
        concat = Image.new(inp_pil.mode, (w * 2, h))
        concat.paste(inp_pil, (0, 0))
        concat.paste(out_pil, (w, 0))
        concat.save(output_dir / fname[0])

if __name__ == "__main__":
    main()
