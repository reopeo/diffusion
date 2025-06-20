import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from omegaconf import OmegaConf
from scipy import integrate

from model import Unet

eps = 1e-3

class PairDataset(Dataset):
    def __init__(self, pt_dir):
        self.pt_dir = Path(pt_dir)
        self.x0_files = sorted(self.pt_dir.glob("noise_*.pt"))

    def __len__(self):
        return len(self.x0_files)

    def __getitem__(self, idx):
        x0_path = self.x0_files[idx]
        idx_str = x0_path.stem.split("_")[1]
        x1_path = self.pt_dir / f"sample_{idx_str}.pt"
        x0 = torch.load(x0_path)
        x1 = torch.load(x1_path)
        return x0, x1


@torch.no_grad()
def sample_ode(model, image_size, batch_size=16, channels=1):
    shape = (batch_size, channels, image_size, image_size)
    device = next(model.parameters()).device

    b = shape[0]
    x = torch.randn(shape, device=device)
    
    def ode_func(t, x):
        x = torch.tensor(x, device=device, dtype=torch.float).reshape(shape)
        t = torch.full(size=(b,), fill_value=t, device=device, dtype=torch.float).reshape((b,))
        v = model(x, t)
        return v.cpu().numpy().reshape((-1,)).astype(np.float64)
    
    res = integrate.solve_ivp(ode_func, (eps, 1.), x.reshape((-1,)).cpu().numpy(), method='RK45')
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape)
    return x.clamp(-1, 1)

def loss_fn(model, x0, x1, t):
    # x_t = t * x1 + (1 - t) * x0
    x_t = t[:, None, None, None] * x1 + (1 - t[:, None, None, None]) * x0
    v = model(x_t, t)
    return F.mse_loss(x1 - x0, v)


def main():
    parser = ArgumentParser(description="Reflow 学習: x0,x1 ペアからの学習スクリプト")
    parser.add_argument('--config',  type=str, required=True, help="OmegaConf YAML のパス")
    parser.add_argument('--pt_dir',  type=str, required=True, help="x0,x1 pt ファイルのディレクトリ")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    torch.manual_seed(42)

    ds = PairDataset(args.pt_dir)
    dl = DataLoader(ds,
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=config.num_workers if 'num_workers' in config else 4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet(**config.model).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr if 'lr' in config else 1e-4)

    for epoch in range(1, config.epochs + 1):
        losses = []
        bar = tqdm(dl, desc=f'Epoch {epoch}')
        for x0, x1 in bar:
            x0 = x0.to(device)
            x1 = x1.to(device)
            batch_size = x0.shape[0]

            t = torch.empty(batch_size, device=device).uniform_(eps, 1)

            loss = loss_fn(model, x0, x1, t)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            losses.append(loss.item())
            bar.set_postfix_str(f'Loss: {np.mean(losses):.6f}')

        if epoch % config.image_interval == 0:
            images = sample_ode(model, config.img_size, channels=config.model_channels)
            img = make_grid((images + 1) / 2, nrow=4)  # [-1,1] → [0,1]
            img_dir = Path(config.output_dir) / 'images'
            img_dir.mkdir(parents=True, exist_ok=True)
            save_image(img, img_dir / f'epoch_{epoch}.png')

        if epoch % config.ckpt_interval == 0:
            ckpt_dir = Path(config.output_dir) / 'ckpt'
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt_dir / f'epoch_{epoch:05d}.pth')

if __name__ == '__main__':
    main()

# python3 train_reflow.py \
#   --config /home/aisl/smrt/diffusion/rect_flow/configs/mnist_reflow.yaml \
#   --pt_dir /home/aisl/smrt/diffusion/rect_flow/generate/01/pt_pairs
