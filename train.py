"""Train the brain-tumor classifier (ViT main model, CNN baseline).

ডেটাসেট: Kaggle Brain Tumor MRI Dataset — data/ ফোল্ডারে থাকতে হবে:
    data/Training/glioma|meningioma|notumor|pituitary/...
    data/Testing/glioma|meningioma|notumor|pituitary/...
(নামাতে: python download_dataset.py — ৭,০২৩ ছবি, ~১৫৬ MB)

Google Colab-এ GPU দিয়ে চালানো সুপারিশ করা (৭,০২৩ ছবি):
    !pip install -r requirements.txt
    !python download_dataset.py
    !python train.py --epochs 15 --model vit --batch 32

নিজের পিসিতে (CPU-তে ধীর হবে, তবু চলবে):
    python train.py --epochs 5 --model cnn

সেভ হয়: models/best_model.pth + models/metrics.json
(accuracy, per-class precision/recall/F1, confusion matrix)

Augmentation-এ মোবাইল-ছবির কথা ভাবা হয়েছে: rotation / perspective /
brightness / blur — যাতে MRI ফিল্ম বা স্ক্রিনের মোবাইলে তোলা ছবিও
চিনতে পারে। এতে মোবাইলের ছবি ডিজিটাল ফাইলের সমান ভালো হবে না —
রিপোর্টে দুই রকম ছবির ফল আলাদা করে লিখতে হবে।
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CLASSES, build_model

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"


def get_train_transforms():
    """ট্রেনিংয়ের ছবি 224x224 করে, augmentation দেয়।"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        # মোবাইল-ছবির জন্য: একটু বাঁকা / আলো কম / ঝাপসা ছবি
        transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.25, contrast=0.25),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_test_transforms():
    """টেস্টের ছবিতে augmentation নেই — শুধু resize + normalize।"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pred, all_true = [], []
    for images, labels in loader:
        outputs = model(images.to(device))
        all_pred += outputs.argmax(1).cpu().tolist()
        all_true += labels.tolist()
    acc = sum(p == t for p, t in zip(all_pred, all_true)) / len(all_true)
    return acc, all_true, all_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit", choices=["vit", "cnn"],
                    help="vit = Vision Transformer (মূল), cnn = ResNet50 বেসলাইন")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if not (DATA_DIR / "Training").exists():
        raise FileNotFoundError(
            "data/Training পাওয়া যায়নি। প্রথমে চালাও: python download_dataset.py")

    train_ds = datasets.ImageFolder(DATA_DIR / "Training", get_train_transforms())
    test_ds = datasets.ImageFolder(DATA_DIR / "Testing", get_test_transforms())
    print("Train:", len(train_ds), "| Test:", len(test_ds), "| Classes:", CLASSES)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch, num_workers=2)

    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    MODELS_DIR.mkdir(exist_ok=True)
    best_acc = 0.0
    best_path = MODELS_DIR / "best_model.pth"
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        test_acc, _, _ = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{args.epochs} | loss {loss:.4f} | "
              f"train {train_acc:.4f} | test {test_acc:.4f} | {time.time() - t0:.0f}s")
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_path)
            print("  -> saved best model:", best_path)

    # সেরা মডেল দিয়ে চূড়ান্ত হিসাব
    model.load_state_dict(torch.load(best_path, map_location=device))
    acc, y_true, y_pred = evaluate(model, test_loader, device)
    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "model": args.model, "test_accuracy": acc, "best_accuracy": best_acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "classes": CLASSES,
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nFinal test accuracy:", round(acc, 4))
    print("Precision / Recall / F1 (per class):")
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    print("Confusion matrix (rows=true, cols=predicted):\n", cm)
    print("Metrics saved to models/metrics.json")


if __name__ == "__main__":
    main()
