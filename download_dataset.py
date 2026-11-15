"""Download the Kaggle Brain Tumor MRI Dataset into data/.

ডেটাসেট: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
মোট ৭,০২৩ ছবি, ৪ ক্লাস: glioma ১,৬২১ · meningioma ১,৬৪৫ ·
pituitary ১,৭৫৭ · no tumor ২,০০০। Training ৫,৭১২ / Testing ১,৩১১।
সাইজ ~১৫৬ MB, CC0 লাইসেন্স (ফ্রি ব্যবহার করা যায়)।

চালানোর আগে (একবারই লাগবে) — দুইটা উপায়, যেকোনো একটা:
  উপায় A (লগইন ছাড়া, সুপারিশ করা):
    python download_dataset.py --mirror
    — Hugging Face থেকে 236 MB zip নামিয়ে data/ ফোল্ডারে সাজাবে
      (9,040 ছবি, train 8,306 / val 488 / test 246 — 2026-11-13 যাচাই করা)
  উপায় B (Kaggle):
  ১. kaggle.com-এ ফ্রি অ্যাকাউন্ট খোলো।
  ২. টার্মিনালে:  pip install kaggle
  ৩. Kaggle সাইটের Account > API থেকে kaggle.json নামাও,
     আর রাখো:  ~/.kaggle/kaggle.json   (Windows হলে C:\\Users\\<নাম>\\.kaggle\\)

তারপর চালাও:
    python download_dataset.py            # Kaggle উপায়
    python download_dataset.py --mirror   # লগইন ছাড়া Hugging Face উপায়

এই স্ক্রিপ্ট ডাউনলোড করে না — এটা চালালেই ডাউনলোড হবে।
ফল: data/Training/<class>/... এবং data/Testing/<class>/...
"""
import argparse
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

DATASET = "masoudnickparvar/brain-tumor-mri-dataset"
MIRROR_URL = ("https://huggingface.co/datasets/usamaJabar/"
              "brain-tumor-mri-classification-merged/resolve/main/"
              "brain-tumor-mri-classification.zip")
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"


def link_mirror():
    """Hugging Face mirror-এর images/ ফোল্ডারকে data/Training|Testing-এ লিংক করা।"""
    src = DATA_DIR / "merged_dataset_v3" / "images"
    for split, target in (("train", "Training"), ("test", "Testing")):
        out = DATA_DIR / target
        out.mkdir(parents=True, exist_ok=True)
        for cls, link_name in (("glioma", "glioma"), ("meningioma", "meningioma"),
                               ("no_tumor", "notumor"), ("pituitary", "pituitary")):
            link = out / link_name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(src / split / cls, target_is_directory=True)


def download_mirror():
    print("Mirror:", MIRROR_URL)
    print("ধাপ ১: Hugging Face থেকে ডাউনলোড করা হচ্ছে (~236 MB, লগইন লাগে না)...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zf_path = DATA_DIR / "brain-tumor-mri-classification.zip"
    urllib.request.urlretrieve(MIRROR_URL, zf_path)
    print("ধাপ ২: images/ ফোল্ডারটুকু খোলা হচ্ছে...")
    with zipfile.ZipFile(zf_path) as z:
        for name in z.namelist():
            if "/images/" in name and name.endswith((".jpg", ".jpeg", ".png")):
                z.extract(name, DATA_DIR)
    link_mirror()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", action="store_true",
                    help="Kaggle-এর বদলে Hugging Face mirror থেকে নামাও (লগইন লাগে না)")
    args = ap.parse_args()
    if args.mirror:
        download_mirror()
    else:
        print("Dataset:", DATASET)
        print("Target folder:", DATA_DIR)
        print("ধাপ ১: Kaggle API দিয়ে ডাউনলোড করা হচ্ছে (~১৫৬ MB)...")
        print("  (kaggle.json না থাকলে আগে উপরের নির্দেশনা দেখো —")
        print("   অথবা লগইন ছাড়া চালাও: python download_dataset.py --mirror)")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "kaggle", "datasets", "download",
               "-d", DATASET, "-p", str(DATA_DIR), "--unzip"]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("\nKaggle API দিয়ে হয়নি। বিকল্প: kagglehub ব্যবহার করো:")
            print("  pip install kagglehub")
            print("  python -c \"import kagglehub; "
                  "print(kagglehub.dataset_download('"
                  + DATASET + "'))\"")
            print("তারপর সেই ফোল্ডার থেকে Training/Testing data/ ফোল্ডারে কপি করো।")
            print("অথবা সহজ উপায়: python download_dataset.py --mirror")
            sys.exit(result.returncode)

        # --unzip না চললে zip ফাইল নিজে খুলে দেওয়া
        for zf in DATA_DIR.glob("*.zip"):
            print("Unzipping:", zf)
            with zipfile.ZipFile(zf) as z:
                z.extractall(DATA_DIR)
            zf.unlink()

    print("\nডাউনলোড শেষ। যাচাই করো:")
    for split in ("Training", "Testing"):
        for cls in ("glioma", "meningioma", "notumor", "pituitary"):
            folder = DATA_DIR / split / cls
            n = len(list(folder.glob("*.*"))) if folder.exists() else 0
            print(f"  {split}/{cls}: {n} images")
    print("\nএরপর চালাও:  python train.py --epochs 15 --model vit")


if __name__ == "__main__":
    main()
