
"""NeuroScan AI - Brain Tumor Detection using ResNet18"""

from pathlib import Path

import cv2
import numpy as np
import torch

from flask import Flask, render_template, request
from PIL import Image

from model import CLASSES, build_model, gradcam_heatmap
from train import get_test_transforms


ROOT = Path(__file__).parent
STATIC_DIR = ROOT / "static"
STATIC_DIR.mkdir(exist_ok=True)

MODEL_PATH = ROOT / "models" / "best_model_resnet18.pth"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    """Load the trained ResNet18 model."""

    global _model

    if _model is None:

        if not MODEL_PATH.exists():
            print("ResNet18 model file not found!")
            return None

        print("Loading ResNet18 model...")

        _model = build_model("resnet18", pretrained=False)

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=_device
        )

        # Support both normal state_dict and checkpoint dictionary
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        _model.load_state_dict(checkpoint)

        _model.to(_device)
        _model.eval()

        print("ResNet18 model loaded successfully!")

    return _model


def clean_phone_photo(img: Image.Image) -> Image.Image:
    """Crop the MRI scan area from a phone photo."""

    arr = np.array(img.convert("L"))

    _, thresh = cv2.threshold(
        arr, 30, 255, cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        x, y, w, h = cv2.boundingRect(
            max(contours, key=cv2.contourArea)
        )

        if w > 50 and h > 50:
            img = img.crop((x, y, x + w, y + h))

    return img


def save_heatmap_overlay(image, heat, out_path):
    """Save Grad-CAM heatmap overlay."""

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    base = np.array(
        image.resize((224, 224))
    ).astype(np.float32) / 255.0

    heat_array = heat.detach().cpu().numpy()

    heat_array = np.squeeze(heat_array)

    heat_array = cv2.resize(
        heat_array.astype(np.float32),
        (224, 224)
    )

    if heat_array.max() > heat_array.min():
        heat_array = (
            heat_array - heat_array.min()
        ) / (
            heat_array.max() - heat_array.min()
        )

    heatmap = plt.get_cmap("jet")(heat_array)[:, :, :3]

    overlay = np.clip(
        base * 0.5 + heatmap * 0.5,
        0,
        1
    )

    plt.imsave(str(out_path), overlay)


@app.route("/", methods=["GET"])
def home():

    ready = MODEL_PATH.exists()

    return render_template(
        "index.html",
        ready=ready,
        classes=CLASSES
    )


@app.route("/predict", methods=["POST"])
def predict():

    model = load_model()

    if model is None:

        return render_template(
            "index.html",
            ready=False,
            classes=CLASSES,
            error="ResNet18 model file not found. Check models folder."
        )

    if "file" not in request.files or request.files["file"].filename == "":

        return render_template(
            "index.html",
            ready=True,
            classes=CLASSES,
            error="Please select an MRI image."
        )

    try:

        uploaded_file = request.files["file"]

        image = Image.open(uploaded_file).convert("RGB")

        image = clean_phone_photo(image)

        transform = get_test_transforms()

        x = transform(image).unsqueeze(0).to(_device)

        with torch.no_grad():

            probs = torch.softmax(model(x), dim=1)[0]

        idx = int(probs.argmax())

        # Grad-CAM is optional. Prediction can still work if
        # the Grad-CAM implementation is not compatible.
        heatmap_filename = None

        try:

            heat = gradcam_heatmap(
                model,
                x,
                target_class=idx
            )

            heatmap_path = STATIC_DIR / "heatmap.jpg"

            save_heatmap_overlay(
                image,
                heat,
                heatmap_path
            )

            heatmap_filename = "heatmap.jpg"

        except Exception as gradcam_error:

            print("Grad-CAM warning:", gradcam_error)

        result = {
            "predicted": CLASSES[idx],

            "confidence": round(
                float(probs[idx]) * 100, 2
            ),

            "all_probabilities": {
                c: round(float(p) * 100, 2)
                for c, p in zip(CLASSES, probs)
            }
        }

        return render_template(
            "index.html",
            ready=True,
            classes=CLASSES,
            result=result,
            heatmap=heatmap_filename
        )

    except Exception as e:

        print("Prediction error:", e)

        return render_template(
            "index.html",
            ready=True,
            classes=CLASSES,
            error=f"Prediction failed: {str(e)}"
        )


@app.route("/report", methods=["GET"])
def report():

    return (
        "<h2>PDF Report</h2>"
        "<p>PDF report feature will be available in Phase 2.</p>"
        "<p><a href='/'>Back to Home</a></p>"
    )


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
