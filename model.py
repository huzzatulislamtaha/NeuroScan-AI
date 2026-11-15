"""Model definitions: Vision Transformer (main) + CNN baseline.

মূল মডেল: ViT-Base (vit_base_patch16_224), ImageNet-এ pretrained,
শেষের classification head বদলে আমাদের ৪ ক্লাসের জন্য বানানো।
বেসলাইন: ResNet50 — রিপোর্টে দেখানোর জন্য যে ViT একই ডেটায়
সাধারণ CNN-এর চেয়ে ভালো করে (আর ViT ভারী লাগলে বিকল্প হিসেবেও)।

ক্লাসের ক্রম (Kaggle ফোল্ডারের alphabetical ক্রমের সাথে মিলিয়ে):
    0 = glioma, 1 = meningioma, 2 = notumor, 3 = pituitary
ImageFolder এই ক্রমেই লেবেল দেয়, তাই এই তালিকা বদলাবে না।

Grad-CAM: `gradcam_heatmap()` দেখো — ResNet50-এর জন্য আসল Grad-CAM,
ViT-এর জন্য gradient-saliency heatmap (ViT-এ conv layer নেই বলে)।
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ক্লাসের নাম — Kaggle ডেটাসেটের ফোল্ডারের নামের সাথে মিলিয়ে রাখা
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASSES)


def build_vit(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """Vision Transformer (timm). `timm` লাগবে — দেখো requirements.txt।"""
    import timm

    model = timm.create_model(
        "vit_base_patch16_224", pretrained=pretrained, num_classes=num_classes
    )
    return model


def build_cnn_baseline(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """ResNet50 বেসলাইন — শেষের fc layer ৪ ক্লাসের জন্য বদলানো।"""
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_resnet18(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """ResNet18 — তুলনার জন্য বেসলাইন (আগের FINAL মডেল)।

    ViT-ই এখন FINAL মডেল (Colab T4 GPU-তে ট্রেন করা, ১০০%)।
    ResNet18 (95.93%) শুধু রিপোর্টে তুলনা দেখাতে রাখা হয়েছে;
    app.py / predict.py এখন ViT লোড করে।"""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(name: str = "vit", num_classes: int = NUM_CLASSES,
                pretrained: bool = True):
    """name='vit' (মূল মডেল) অথবা name='cnn' (ResNet50 বেসলাইন)।"""
    if name == "vit":
        return build_vit(num_classes, pretrained)
    if name == "cnn":
        return build_cnn_baseline(num_classes, pretrained)
    if name == "resnet18":
        return build_resnet18(num_classes, pretrained)
    raise ValueError(f"Unknown model '{name}'. Use 'vit', 'cnn' or 'resnet18'.")


def gradcam_heatmap(model, image_tensor, target_class=None):
    """একটা ছবির জন্য heatmap বানায় (224x224, 0-1 মানের tensor)।

    - ResNet50 হলে: শেষ conv block (layer4)-এর ওপর আসল Grad-CAM।
    - ViT হলে: input-এর সাপেক্ষে class score-এর gradient থেকে
      saliency heatmap (ViT-এ conv layer নেই, তাই এটাই সঠিক পদ্ধতি)।
    image_tensor: (1, 3, 224, 224) আকারের tensor হতে হবে।
    ফেরত: (224, 224) আকারের tensor, মান 0 থেকে 1-এর মধ্যে।
    """
    model.eval()
    x = image_tensor.clone().detach()

    # --- ResNet50 (CNN) হলে আসল Grad-CAM ---
    if hasattr(model, "layer4"):
        activations, gradients = [], []

        def save_activation(_m, _i, out):
            activations.append(out)

        def save_gradient(_m, _gi, go):
            gradients.append(go[0])

        handle_f = model.layer4.register_forward_hook(save_activation)
        handle_b = model.layer4.register_full_backward_hook(save_gradient)
        try:
            output = model(x)
            if target_class is None:
                target_class = int(output.argmax(dim=1).item())
            model.zero_grad()
            output[0, target_class].backward()
        finally:
            handle_f.remove()
            handle_b.remove()

        act = activations[0][0]          # (C, H, W)
        grad = gradients[0][0]           # (C, H, W)
        weights = grad.mean(dim=(1, 2))  # প্রতিটি channel-এর গুরুত্ব
        heat = (weights[:, None, None] * act).sum(dim=0)
        heat = F.relu(heat)
        heat = F.interpolate(
            heat[None, None], size=(224, 224),
            mode="bilinear", align_corners=False,
        )[0, 0]
    else:
        # --- ViT হলে gradient saliency ---
        x.requires_grad_(True)
        output = model(x)
        if target_class is None:
            target_class = int(output.argmax(dim=1).item())
        model.zero_grad()
        output[0, target_class].backward()
        heat = x.grad[0].abs().mean(dim=0)  # (224, 224)
        heat = F.relu(heat)

    heat = heat - heat.min()
    if heat.max() > 0:
        heat = heat / heat.max()
    return heat.detach()


if __name__ == "__main__":
    # দ্রুত shape পরীক্ষা (ট্রেনিং ডেটা লাগে না):  python model.py
    for name in ("vit", "cnn"):
        m = build_model(name, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        y = m(x)
        print(name, "->", tuple(y.shape), "(batch, 4 classes)")
        # Grad-CAM-ও পরীক্ষা করা যায়
        h = gradcam_heatmap(m, torch.randn(1, 3, 224, 224))
        print(name, "heatmap ->", tuple(h.shape))
