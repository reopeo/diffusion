import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms as T
from omegaconf import OmegaConf

from model import Unet

eps = 1e-3

@torch.no_grad()
def sample_euler(model, image_size, batch_size=16, channels=1, steps=1000, device='cpu'):
    shape = (batch_size, channels, image_size, image_size)
    x = torch.randn(shape, device=device)
    t_vals = torch.linspace(eps, 1.0, steps, device=device)
    dt = t_vals[1] - t_vals[0]

    for t in t_vals:
        t_tensor = torch.full((batch_size,), fill_value=t, device=device)
        v = model(x, t_tensor)
        x = x + dt * v

    return x.clamp(-1, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     type=str, required=True)
    parser.add_argument('--ckpt',       type=str, required=True)
    parser.add_argument('--num_images', type=int, default=16)
    parser.add_argument('--steps',      type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--out_dir',    type=str, default='samples')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet(**config.model).to(device).eval()

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])
    start_epoch = ckpt.get('epoch', '?')
    print(f"Loaded checkpoint from epoch {start_epoch}")

    out_root = Path(args.out_dir)
    png_dir = out_root / 'class0'
    # pil_dir = out_root / 'pil'
    png_dir.mkdir(parents=True, exist_ok=True)
    # pil_dir.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size or args.num_images
    total = args.num_images
    generated = 0
    to_pil = T.ToPILImage()

    while generated < total:
        n = min(batch_size, total - generated)
        imgs = sample_euler(
            model,
            image_size=config.img_size,
            batch_size=n,
            channels=config.model.channels,
            steps=args.steps,
            device=device
        )
        imgs = (imgs + 1) / 2

        for i in range(n):
            idx = generated + i
            pil_img = to_pil(imgs[i])
            pil_img.save(png_dir / f'{idx:04d}.png')
            # with open(pil_dir / f'{idx:04d}.pkl', 'wb') as f:
            #     pickle.dump(pil_img, f)

        generated += n
        print(f"Saved {generated}/{total} images")

if __name__ == '__main__':
    main()

# python generate.py \
#   --config configs/mydataset.yaml \
#   --ckpt outputs/mydataset/ckpt/epoch_00100.pth \
#   --num_images 25 \
#   --steps 500 \
#   --out_dir generated_samples
