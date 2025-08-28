import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms as T
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from pathlib import Path
from scipy import integrate
from argparse import ArgumentParser
from omegaconf import OmegaConf
from PIL import Image

from model import Unet

eps = 1e-3

@torch.no_grad()
def sample_ode(model, image_size, batch_size=16, channels=1, x_init=None):
    shape = (batch_size, channels, image_size, image_size)
    device = next(model.parameters()).device

    b = shape[0]
    if x_init is not None:
        x = x_init.to(device)
    else:
        x = torch.randn(shape, device=device)
    
    def ode_func(t, x):
        x = torch.tensor(x, device=device, dtype=torch.float).reshape(shape)
        t = torch.full(size=(b,), fill_value=t, device=device, dtype=torch.float).reshape((b,))
        v = model(x, t)
        return v.cpu().numpy().reshape((-1,)).astype(np.float64)
    
    res = integrate.solve_ivp(ode_func, (eps, 1.), x.reshape((-1,)).cpu().numpy(), method='RK45')
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape)
    return x.clamp(-1, 1)

def random_mask(img, mask_ratio=0.5):
    # img: Tensor (C,H,W), mask_ratio: 0~1
    mask = torch.rand_like(img) > mask_ratio
    return img * mask.float()

def loss_fn(model, x_dense, x_sparse, t):
    # x_dense: target, x_sparse: input
    x_0 = x_sparse
    x_1 = x_dense
    x_t = t[:, None, None, None] * x_1 + (1 - t[:, None, None, None]) * x_0
    v = model(x_t, t)
    loss = F.mse_loss(x_1 - x_0, v)
    return loss

class SparseDenseImageDataset(Dataset):
    def __init__(self, sparse_dir, dense_dir, transform, img_convert="RGB"):
        self.sparse_dir = Path(sparse_dir)
        self.dense_dir = Path(dense_dir)
        self.transform = transform
        self.img_convert = img_convert
        # 画像ファイル名のリスト（両方に存在するもののみ）
        sparse_files = set(f.name for f in self.sparse_dir.glob("*"))
        dense_files = set(f.name for f in self.dense_dir.glob("*"))
        self.filenames = sorted(list(sparse_files & dense_files))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        sparse_img = Image.open(self.sparse_dir / fname).convert(self.img_convert)
        dense_img = Image.open(self.dense_dir / fname).convert(self.img_convert)
        sparse = self.transform(sparse_img)
        dense = self.transform(dense_img)
        return {"sparse": sparse, "dense": dense}

class ToMaxPooledTensor:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        tensor = T.ToTensor()(img)
        c, h, w = tensor.shape
        kh = h // self.target_size
        kw = w // self.target_size
        # MaxPool2d expects (N, C, H, W), so add batch dim
        tensor = tensor.unsqueeze(0)
        pool = torch.nn.MaxPool2d(kernel_size=(kh, kw), stride=(kh, kw))
        pooled = pool(tensor)
        pooled = pooled.squeeze(0)
        # If pooled size is not exactly target_size, crop or pad as needed
        pooled = pooled[:, :self.target_size, :self.target_size]
        return (pooled * 2) - 1

def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--sparse_dir', type=str, required=True)
    parser.add_argument('--dense_dir', type=str, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    torch.manual_seed(42)

    output_dir = Path(config.output_dir)
    img_dir = output_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / 'ckpt'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    transform = ToMaxPooledTensor(config.img_size)

    dataset = SparseDenseImageDataset(
        args.sparse_dir, args.dense_dir, transform, getattr(config, "img_convert", "RGB")
    )
    dl = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet(**config.model).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    def loss_fn(model, x_dense, x_sparse, t):
        # x_dense: target, x_sparse: input
        x_0 = x_sparse
        x_1 = x_dense
        x_t = t[:, None, None, None] * x_1 + (1 - t[:, None, None, None]) * x_0
        v = model(x_t, t)
        loss = F.mse_loss(x_1 - x_0, v)
        return loss

    def handle_batch(batch):
        batch_size = batch["dense"].shape[0]
        x_dense = batch["dense"].to(device)
        x_sparse = batch["sparse"].to(device)

        t = torch.empty(size=(batch_size,), device=device).uniform_(eps, 1)
        loss = loss_fn(model, x_dense, x_sparse, t)
        return loss

    # DataLoaderのイテレータを作成
    dl_iter = iter(dl)

    train_losses = list()
    for epoch in range(1, config.epochs + 1):
        losses = list()
        bar = tqdm(dl, total=len(dl), desc=f'Epoch {epoch}: ')
        for batch in bar:
            optimizer.zero_grad()
            loss = handle_batch(batch)
            loss.backward()
            clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())
            bar.set_postfix_str(f'Loss: {np.mean(losses):.6f}')
        train_losses.append(np.mean(losses))
        if epoch % config.image_interval == 0:
            try:
                batch = next(dl_iter)
            except StopIteration:
                dl_iter = iter(dl)
                batch = next(dl_iter)
            x_sparse = batch["sparse"][:4]  # 例: 4枚だけ可視化
            images = sample_ode(
                model, config.img_size, batch_size=x_sparse.shape[0], channels=config.model.channels, x_init=x_sparse
            )
            img = make_grid(images, nrow=2, normalize=True)
            img = T.ToPILImage()(img)
            img.save(img_dir / f'epoch_{epoch}.png')

        if epoch % config.ckpt_interval == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt_dir / f'epoch_{epoch:05d}.pth')


if __name__ == '__main__':
    main()
