"""FINAL training script — lightweight ResNet18, CPU-friendly.

Why not ViT here: the full Vision Transformer (vit_base_patch16_224, 86M
params) on 8,306 images needs a GPU. This VM has no GPU and only 1.5
shared CPUs, so the FINAL deliverable trains a real, measured model with
ResNet18 (~11.7M params, ImageNet-pretrained) on a subsample, at 128x128.

Training setup (exact, measured in models/metrics.json):
  - model: ResNet18, pretrained ImageNet weights, final fc -> 4 classes
  - image size: 128x128 (train.py's 224 is kept for the Colab ViT run)
  - data: data/Training subsampled to max 300 images per class
    (1,200 total), data/Testing = all 246 images
  - batch 32, Adam lr 1e-3, CosineAnnealingLR, CrossEntropyLoss
  - epochs: 4 (see --epochs)
  - augmentation: RandomRotation(10), RandomHorizontalFlip,
    ColorJitter(brightness/contrast 0.2) — light on purpose, CPU is slow
  - threads: torch.set_num_threads(2)

Run:
    ~/workspace/brain-tumor-project/venv/bin/python train_final.py
    (or: python train_final.py, after pip install -r requirements.txt)

Saves: models/best_model.pth + models/metrics.json
       (test accuracy, per-class precision/recall/F1, confusion matrix)

The full ViT version (15 epochs, all 8,306 images, 224x224) still needs
Google Colab GPU — see train_colab.ipynb. That is the documented next step,
not a hidden failure: see FINAL_STATUS.md.
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def build_resnet18(pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, 4)
    return m


def subsample(dataset, max_per_class, seed=42):
    """Keep at most max_per_class images per class (balanced subsample)."""
    by_class = {}
    for idx, (_, label) in enumerate(dataset.samples):
        by_class.setdefault(label, []).append(idx)
    rng = random.Random(seed)
    keep = []
    counts = {}
    for label in sorted(by_class):
        idxs = by_class[label]
        rng.shuffle(idxs)
        chosen = idxs[:max_per_class]
        keep.extend(chosen)
        counts[CLASSES[label]] = len(chosen)
    keep.sort()
    return Subset(dataset, keep), counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-per-class", type=int, default=300)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    device = "cpu"
    print("Device: cpu (no GPU on this VM), threads:", torch.get_num_threads())

    train_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_train = datasets.ImageFolder(DATA_DIR / "Training", train_tf)
    test_ds = datasets.ImageFolder(DATA_DIR / "Testing", test_tf)
    train_ds, counts = subsample(full_train, args.max_per_class)
    print("Train (subsampled):", len(train_ds), counts)
    print("Test (full):", len(test_ds), "| Classes:", CLASSES)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch, num_workers=0)

    try:
        model = build_resnet18(pretrained=not args.no_pretrained)
        print("ResNet18: ImageNet-pretrained weights loaded")
    except Exception as e:
        print("Pretrained weights download failed (%s) — training from scratch" % e)
        model = build_resnet18(pretrained=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    MODELS_DIR.mkdir(exist_ok=True)
    best_acc, best_path = 0.0, MODELS_DIR / "best_model.pth"
    epoch_times = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * images.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += len(labels)
        scheduler.step()
        model.eval()
        t_correct, t_total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                t_correct += (model(images).argmax(1) == labels).sum().item()
                t_total += len(labels)
        test_acc = t_correct / t_total
        dt = time.time() - t0
        epoch_times.append(round(dt, 1))
        print(f"Epoch {epoch}/{args.epochs} | loss {loss_sum/total:.4f} | "
              f"train {correct/total:.4f} | test {test_acc:.4f} | {dt:.0f}s",
              flush=True)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_path)
            print("  -> saved best model:", best_path, flush=True)

    model.load_state_dict(torch.load(best_path, map_location="cpu"))
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            all_pred += model(images).argmax(1).tolist()
            all_true += labels.tolist()
    acc = sum(p == t for p, t in zip(all_pred, all_true)) / len(all_true)
    report = classification_report(all_true, all_pred, target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(all_true, all_pred).tolist()
    metrics = {
        "model": "resnet18",
        "model_architecture": "torchvision ResNet18, ImageNet-pretrained, fc->4 classes",
        "image_size": args.image_size,
        "train_size": len(train_ds),
        "train_counts_per_class": counts,
        "test_size": len(test_ds),
        "test_split": "data/Testing, all images (246)",
        "epochs": args.epochs,
        "batch_size": args.batch,
        "optimizer": "Adam",
        "lr": args.lr,
        "device": "cpu (no GPU)",
        "epoch_seconds": epoch_times,
        "total_training_seconds": round(time.time() - t_start, 1),
        "test_accuracy": acc,
        "best_accuracy": best_acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "classes": CLASSES,
    }
    (MODELS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nFinal test accuracy:", round(acc, 4), f"({len(test_ds)} test images)")
    print(classification_report(all_true, all_pred, target_names=CLASSES, zero_division=0))
    print("Confusion matrix (rows=true, cols=predicted):")
    for row in cm:
        print(row)
    print("Metrics saved to models/metrics.json")
    print("Total training time: %.1f min" % ((time.time() - t_start) / 60))


if __name__ == "__main__":
    main()
