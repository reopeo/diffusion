import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import make_grid, save_image
from omegaconf import OmegaConf

from model import Unet

eps = 1e-3

@torch.no_grad()
def sample_euler(model, image_size, batch_size, channels, steps, device):
    shape = (batch_size, channels, image_size, image_size)
    x0 = torch.randn(shape, device=device)
    x = x0.clone()

    t_vals = torch.linspace(eps, 1.0, steps, device=device)
    if steps == 1:
        dt = 1.0 - eps
    else:
        dt = t_vals[1] - t_vals[0]

    for t in t_vals:
        t_tensor = torch.full((batch_size,), fill_value=t, device=device)
        v = model(x, t_tensor)
        x = x + dt * v

    x1 = x.clamp(-1, 1)
    return x0, x1


def main():
    parser = argparse.ArgumentParser(description="Generate and save (x0, x1) pairs plus PNG preview")
    parser.add_argument('--config',     type=str, required=True, help="OmegaConf YAML")
    parser.add_argument('--ckpt',       type=str, required=True, help="モデルチェックポイント (.pth)")
    parser.add_argument('--num_images', type=int, default=16,    help="生成する総数")
    parser.add_argument('--steps',      type=int, default=1000,  help="Euler 刻みステップ数")
    parser.add_argument('--batch_size', type=int, default=None,  help="バッチサイズ（未指定なら num_images）")
    parser.add_argument('--out_dir',    type=str, default='outputs', help="出力ディレクトリ")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Unet(**config.model).to(device)
    model.eval()
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    out_dir   = Path(args.out_dir)
    pt_dir    = out_dir / 'pt_pairs'
    img_dir   = out_dir / 'images'
    pt_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size or args.num_images
    total      = args.num_images
    generated  = 0
    img_size   = config.img_size
    channels   = config.model.channels

    while generated < total:
        n = min(batch_size, total - generated)
        x0_batch, x1_batch = sample_euler(
            model,
            image_size=img_size,
            batch_size=n,
            channels=channels,
            steps=args.steps,
            device=device
        )

        for i in range(n):
            idx = generated + i
            torch.save(x0_batch[i].cpu(), pt_dir / f'noise_{idx:04d}.pt')
            torch.save(x1_batch[i].cpu(), pt_dir / f'sample_{idx:04d}.pt')

            img = (x1_batch[i] + 1) / 2
            save_image(img, img_dir / f'sample_{idx:04d}.png')

        generated += n
        print(f"Saved {generated}/{total} samples (PT pairs + PNG)")

if __name__ == '__main__':
    main()

# python3 generate.py \
#   --config /home/aisl/smrt/diffusion/rect_flow/configs/mnist.yaml \
#   --ckpt   /home/aisl/smrt/diffusion/rect_flow/out/mnist/ckpt/epoch_00020.pth \
#   --num_images  100 \
#   --steps       10 \
#   --batch_size  8 \
#   --out_dir     /home/aisl/smrt/diffusion/rect_flow/generate
