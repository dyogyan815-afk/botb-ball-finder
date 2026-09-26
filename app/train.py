from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .model import CoordinateModel

class AnnotationDataset(Dataset):
    def __init__(self, path: str):
        self.rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
        self.tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        row = self.rows[i]; image = Image.open(row["image_path"]).convert("RGB")
        w, h = image.size
        target = torch.tensor([row["x"]/w, row["y"]/h, 0.03, 0.03], dtype=torch.float32)
        return self.tf(image), target

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--annotations", required=True); p.add_argument("--output", default="checkpoints/model.pt"); p.add_argument("--epochs", type=int, default=20); p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args(); ds = AnnotationDataset(args.annotations); loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"; model = CoordinateModel().to(device); opt = torch.optim.Adam(model.head.parameters(), lr=1e-4); loss_fn = nn.SmoothL1Loss()
    for epoch in range(args.epochs):
        model.train(); total = 0.0
        for x, target in loader:
            x, target = x.to(device), target.to(device); out = model(x); loss = loss_fn(out[:, :2].sigmoid(), target[:, :2]) + 0.01 * loss_fn(out[:, 2:], target[:, 2:].log())
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); total += float(loss)
        print(f"epoch={epoch+1} loss={total/max(1,len(loader)):.5f}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); torch.save(model.state_dict(), args.output)

if __name__ == "__main__": main()
