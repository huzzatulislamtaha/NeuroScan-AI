"""Predict one MRI image with a trained model (+ Grad-CAM heatmap).

ব্যবহার:
    python predict.py --image path/to/mri.jpg --model vit
    python predict.py --image path/to/mri.jpg --model vit --heatmap out_heatmap.jpg

দরকার: models/best_model.pth (train.py চালিয়ে বানাতে হয়)।
ফল: predicted class, confidence %, সব ক্লাসের probability,
আর --heatmap দিলে heatmap ছবির ফাইল (মডেল কোথায় তাকিয়েছে)।
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # ছবি ফাইলে সেভ করার জন্য, স্ক্রিন লাগবে না
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from model import CLASSES, build_model, gradcam_heatmap
from train import get_test_transforms


def predict(image_path: str, model_name: str = "vit"):
    """একটা MRI ছবির prediction ফেরত দেয় (dict)।"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(model_name, pretrained=False)
    weights = Path(__file__).parent / "models" / "best_model.pth"
    if not weights.exists():
        raise FileNotFoundError(
            f"Trained model not found: {weights}\nRun train.py first.")
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device).eval()

    image = Image.open(image_path).convert("RGB")
    x = get_test_transforms()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    idx = int(probs.argmax())
    return {
        "image": str(image_path),
        "predicted": CLASSES[idx],
        "confidence": round(float(probs[idx]) * 100, 2),
        "all_probabilities": {c: round(float(p) * 100, 2)
                              for c, p in zip(CLASSES, probs)},
    }


def save_heatmap(image_path: str, model_name: str, out_path: str):
    """আসল ছবির ওপর Grad-CAM heatmap বসিয়ে ফাইল সেভ করে।"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(model_name, pretrained=False)
    weights = Path(__file__).parent / "models" / "best_model.pth"
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device).eval()

    image = Image.open(image_path).convert("RGB").resize((224, 224))
    x = get_test_transforms()(image).unsqueeze(0).to(device)
    heat = gradcam_heatmap(model, x).cpu().numpy()
    heatmap = plt.get_cmap("jet")(heat)[:, :, :3]  # লাল = মডেল বেশি তাকিয়েছে
    overlay = np.array(image) / 255.0 * 0.5 + heatmap * 0.5
    plt.imsave(out_path, np.clip(overlay, 0, 1))
    print("Heatmap saved to:", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="MRI ছবির path")
    ap.add_argument("--model", default="vit",
                    choices=["vit", "cnn", "resnet18"],
                    help="vit = FINAL ট্রেইন করা মডেল (ডিফল্ট, Colab ViT ১০০%), "
                         "resnet18/cnn = তুলনার জন্য বেসলাইন")
    ap.add_argument("--heatmap", default=None,
                    help="heatmap ছবি সেভ করার path, যেমন: result_heatmap.jpg")
    args = ap.parse_args()
    result = predict(args.image, args.model)
    print(result)
    if args.heatmap:
        save_heatmap(args.image, args.model, args.heatmap)
