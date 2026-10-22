from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firesight.config import PROJECT_ROOT
from firesight.data.manifest import load_manifest


def save_crop(img, box, out: Path, margin=0.10) -> bool:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * margin)); y1 = max(0, int(y1 - bh * margin))
    x2 = min(w, int(x2 + bw * margin)); y2 = min(h, int(y2 + bh * margin))
    if x2 - x1 < 12 or y2 - y1 < 12:
        return False
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), crop)
    return True


def random_crop(img, rng, out: Path) -> bool:
    h, w = img.shape[:2]
    size = rng.randint(max(32, min(h, w) // 8), max(48, min(h, w) // 2))
    x = rng.randint(0, max(0, w - size)); y = rng.randint(0, max(0, h - size))
    crop = img[y:y + size, x:x + size]
    if crop.size == 0:
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), crop)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a fire/smoke/lookalike second-stage crop verifier on TRAIN only.")
    parser.add_argument("--out-dir", default="data/verifier")
    parser.add_argument("--model-out", default="models/verifier.onnx")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-positive-images", type=int, default=1800)
    parser.add_argument("--max-negative-images", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--reuse-crops", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    out = PROJECT_ROOT / args.out_dir
    counts = Counter()
    if not args.reuse_crops:
        if out.exists():
            shutil.rmtree(out)
        train = load_manifest(PROJECT_ROOT / "data/splits/train.jsonl", root=PROJECT_ROOT)
        positives = [r for r in train if r.labels]
        negatives = [r for r in train if not r.labels]
        rng.shuffle(positives); rng.shuffle(negatives)
        positives = positives[:args.max_positive_images]
        negatives = negatives[:args.max_negative_images]

        scenes = sorted({r.scene for r in positives + negatives})
        rng.shuffle(scenes)
        val_scenes = set(scenes[:max(1, len(scenes) // 8)])

        for rec in positives:
            split = "val" if rec.scene in val_scenes else "train"
            img = cv2.imread(str(PROJECT_ROOT / rec.path))
            if img is None:
                continue
            for i, gt in enumerate(rec.labels[:4]):
                name = f"{rec.id.replace(':','_')}_{i}.jpg"
                if save_crop(img, gt.box.to_list(), out / split / gt.label / name):
                    counts[(split, gt.label)] += 1
        for rec in negatives:
            split = "val" if rec.scene in val_scenes else "train"
            img = cv2.imread(str(PROJECT_ROOT / rec.path))
            if img is None:
                continue
            for i in range(2):
                name = f"{rec.id.replace(':','_')}_{i}.jpg"
                if random_crop(img, rng, out / split / "lookalike" / name):
                    counts[(split, "lookalike")] += 1

        mined = PROJECT_ROOT / "data/negatives/mined"
        meta_files = [mined / "mined.jsonl"]
        low = PROJECT_ROOT / "data/negatives/mined_lowconf/mined.jsonl"
        if low.exists():
            meta_files.append(low)
        for meta_file in meta_files:
            if not meta_file.exists():
                continue
            for line in meta_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                src = PROJECT_ROOT / row["crop"]
                if not src.exists():
                    continue
                split = "val" if rng.random() < 0.12 else "train"
                dst = out / split / "lookalike" / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                counts[(split, "lookalike")] += 1
    print("crop counts", dict(counts), flush=True)
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

    tf_train = transforms.Compose([
        transforms.Resize((144, 144)),
        transforms.RandomResizedCrop(128, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((128, 128)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.ImageFolder(out / "train", transform=tf_train)
    val_ds = datasets.ImageFolder(out / "val", transform=tf_val)
    print("classes", train_ds.classes, "train", len(train_ds), "val", len(val_ds), flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = mobilenet_v3_small(weights=None if args.resume else MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(train_ds.classes))
    model.to(device)
    checkpoint = out / "last.pt"
    start_epoch = 0
    if args.resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        start_epoch = int(state.get("epoch", 0))
    counts_train = Counter(train_ds.targets)
    weights = torch.tensor([len(train_ds) / (len(counts_train) * counts_train[i]) for i in range(len(train_ds.classes))],
                           dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)
    best_acc = 0.0
    history = []
    hist_path = out / "history.jsonl"
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total = correct = 0
        loss_sum = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss) * len(x)
            total += len(x)
            correct += int((logits.argmax(1) == y).sum())
        model.eval()
        v_total = v_correct = 0
        confusion = np.zeros((len(train_ds.classes), len(train_ds.classes)), dtype=int)
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                v_total += len(x)
                v_correct += int((pred == y).sum())
                for a, b in zip(y.cpu().numpy(), pred.cpu().numpy()):
                    confusion[a, b] += 1
        acc = v_correct / max(v_total, 1)
        row = {"epoch": epoch + 1, "train_loss": round(loss_sum / max(total, 1), 4),
               "train_acc": round(correct / max(total, 1), 4), "val_acc": round(acc, 4),
               "confusion": confusion.tolist()}
        history.append(row)
        with hist_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        print(row, flush=True)
        torch.save({"model": model.state_dict(), "epoch": epoch + 1}, checkpoint)
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True))
    model.eval()
    dummy = torch.randn(1, 3, 128, 128, device=device)
    onnx_path = PROJECT_ROOT / args.model_out
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, dummy, str(onnx_path), input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    (out / "metrics.json").write_text(json.dumps({
        "classes": train_ds.classes, "best_val_acc": best_acc, "history": history,
        "counts": {f"{k[0]}:{k[1]}": v for k, v in counts.items()},
        "model": str(onnx_path), "seed": args.seed,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"best_val_acc": best_acc, "onnx": str(onnx_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
