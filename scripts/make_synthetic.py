"""Generates N synthetic grayscale test images (shapes + noise) of mixed sizes."""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

p = argparse.ArgumentParser()
p.add_argument("--out", default="data")
p.add_argument("--count", type=int, default=200)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(0)
for i in range(a.count):
    w, h = (int(v) for v in rng.integers(256, 2049, 2))
    img = Image.fromarray(
        (np.linspace(0, 255, w) * np.ones((h, 1))).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(5, 30))):
        x, y = rng.integers(0, w), rng.integers(0, h)
        r = int(rng.integers(10, 200))
        box, fill = [x, y, x + r, y + r], int(rng.integers(0, 256))
        (d.ellipse if rng.random() < 0.5 else d.rectangle)(box, fill=fill)
    noisy = np.asarray(img, np.float32) + rng.normal(0, 6, (h, w))
    Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8)).save(
        os.path.join(a.out, f"synthetic_{i:04d}.png"))
print(f"wrote {a.count} images to {a.out}")
