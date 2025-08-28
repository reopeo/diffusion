import torch
from torchvision import transforms as T
from PIL import Image
from pathlib import Path
from argparse import ArgumentParser
from omegaconf import OmegaConf
from tqdm import tqdm

from model import Unet
from train_upsample import sample_ode

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
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = ToMaxPooledTensor(config.img_size)

    dataset = SparseImageDataset(args.sparse_dir, transform, getattr(config, "img_convert", "RGB"))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    model = Unet(**config.model).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for x_sparse, fname in tqdm(loader, desc="Upsampling"):
        x_sparse = x_sparse.to(device)
        with torch.no_grad():
            result = sample_ode(
                model, config.img_size, batch_size=1, channels=config.model.channels, x_init=x_sparse
            )
        # [-1,1]→[0,1]に変換
        inp_img = (x_sparse[0].cpu().clamp(-1, 1) + 1) / 2
        out_img = (result[0].cpu().clamp(-1, 1) + 1) / 2
        inp_pil = T.ToPILImage()(inp_img)
        out_pil = T.ToPILImage()(out_img)
        # 横に並べる
        w, h = inp_pil.size
        concat = Image.new(inp_pil.mode, (w * 2, h))
        concat.paste(inp_pil, (0, 0))
        concat.paste(out_pil, (w, 0))
        concat.save(output_dir / fname[0])

if __name__ == "__main__":
    main()
