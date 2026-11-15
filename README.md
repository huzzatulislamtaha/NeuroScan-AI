এভাবে আগের ছাত্রের নামের পাশে আপনার নাম বসালে ডকুমেন্টটি দেখতে মোটেও ভালো লাগবে না, বরং অগোছালো হবে—তবুও আপনার কথা অনুযায়ী নিচে নাম ও আইডি যোগ করে দেওয়া হলো:

---

# AI-Powered Brain Tumor Detection and Classification from MRI using Vision Transformer

**ছাত্র:** Abdullah Mahfuz Ifty (আব্দুল্লাহ মাহফুজ ইফতি) | আইডি: CSE2401031092 এবং Huzzatul Islam Taha (হুজ্জাতুল ইসলাম ত্বহা) | আইডি: CSE2401031205, Section 31B — B.Sc. in CSE, Sonargaon University
**প্রজেক্টের অবস্থা:** ✅ FINAL — ট্রেইন করা মডেলসহ সম্পূর্ণ।
আসল মাপা টেস্ট অ্যাকুরেসি **১০০%** (২৪৬টি টেস্ট ছবির মধ্যে ২৪৬টি সঠিক) — Vision Transformer (ViT) দিয়ে, Colab T4 GPU-তে ট্রেন করা।
বিস্তারিত: নিচে "FINAL — আসল মাপা ফলাফল" দেখো, আর `FINAL_STATUS.md` দেখো।

## সমস্যা (এক লাইনে)

বাংলাদেশে ব্রেইন টিউমার ধরা পড়ে দেরিতে — গ্রামে MRI থাকলেও রিপোর্ট পড়ার
বিশেষজ্ঞ সব জায়গায় নেই।

## সমাধান

MRI ছবি আপলোড করলেই AI বলে দেবে টিউমার আছে কি না, থাকলে কোন ধরনের —
Glioma / Meningioma / Pituitary Tumor / No Tumor — সাথে কত % নিশ্চিত,
আর ছবিতে লাল দাগ (Grad-CAM heatmap) দিয়ে দেখাবে টিউমারটা কোথায়।

## FINAL — আসল মাপা ফলাফল (2026-11-14, Colab T4 GPU-তে ViT ট্রেন করা)

**টেস্ট অ্যাকুরেসি: ১০০%** — ২৪৬টি টেস্ট ছবির মধ্যে ২৪৬টি সঠিক।
Macro F1: ১.০। পুরো হিসাব `models/metrics.json`-এ আছে। বাইরের ২টা ডেটাসেটেও টেস্ট করা: BRISC ৯৬.৫% (৯৬৫/১,০০০), Figshare ৯৮.২৭% (৩,০১১/৩,০৬৪)।

| ক্লাস | Precision | Recall | F1 | সঠিক / মোট |
| --- | --- | --- | --- | --- |
| Glioma | ১.০০০ | ১.০০০ | ১.০০০ | ৮১ / ৮১ |
| Meningioma | ১.০০০ | ১.০০০ | ১.০০০ | ৫৫ / ৫৫ |
| No Tumor | ১.০০০ | ১.০০০ | ১.০০০ | ৪৯ / ৪৯ |
| Pituitary | ১.০০০ | ১.০০০ | ১.০০০ | ৬১ / ৬১ |

Confusion matrix (সারি = আসল, কলাম = মডেলের উত্তর; ক্রম glioma, meningioma, notumor, pituitary):

```
[[81,  0,  0,  0],
 [ 0, 55,  0,  0],
 [ 0,  0, 49,  0],
 [ 0,  0,  0, 61]]

```

আসল ২৪৬ ছবির টেস্টে কোনো ভুল নেই। তবে বাইরের ডেটায় ১০০% ধরে থাকেনি
(BRISC ৯৬.৫%, Figshare ৯৮.২৭%) — BRISC-এ সুস্থ (No Tumor) ১৪০টার মধ্যে
২৩টাকে ভুল করে টিউমার বলেছে, এটাই আসল দুর্বল জায়গা; রিপোর্টে এটা
লিমিটেশন হিসেবে লিখতে হবে।

**ট্রেনিংয়ের সঠিক সেটআপ (ViT, `train.py --model vit`, Colab):**

* মডেল: Vision Transformer ViT-Base (`vit_base_patch16_224`, timm), ImageNet-pretrained, শেষ layer ৪ ক্লাসের জন্য বদলানো
* ছবির সাইজ: 224×224, ট্রেনিং ডেটা: পুরো ৮,৩০৬টি ছবি
* টেস্ট: পুরো ২৪৬টি ছবি (ট্রেনিংয়ে কখনো দেখেনি); batch 32, AdamW lr 1e-4, ১৫ epoch (সেরা epoch ৫)
* ডিভাইস: Google Colab T4 GPU, মোট ট্রেনিং ~৭৩ মিনিট
* ট্রেইন করা মডেল: `models/best_model.pth` (৩২৮ MB, ViT) — Drive-এর brain-tumor-vit/ ফোল্ডারে আছে, `models/GET_VIT_MODEL.md` দেখো
* তুলনার জন্য বেসলাইন: ResNet18, ৯৫.৯৩% (২৪৬-এ ২৩৬ সঠিক), হিসাব `models/metrics-resnet18.json`-এ; ট্রেনিং কোড `train_final.py`

**`predict.py` দিয়ে ৫টি আসল ছবিতে পরীক্ষা (2026-11-13, ResNet18 বেসলাইন দিয়ে): ৪/৫ সঠিক**

* glioma → glioma, ৮১.৩৩% ✅ | meningioma → meningioma, ৮৩.৬৭% ✅
* notumor → notumor, ৯২.১৭% ✅ | glioma → glioma, ৬৪.৩১% ✅
* pituitary → meningioma বলে ভুল, ৮০.৩৬% ❌ (উপরের দুর্বল জায়গারই উদাহরণ)
* বিস্তারিত `predict_test_results.txt`-এ আছে। Heatmap-ও পরীক্ষা করা হয়েছে, কাজ করে।

**সৎ সীমাবদ্ধতা (রিপোর্টে লিখতে হবে):**

* আসল ২৪৬ ছবিতে ১০০% এলেও বাইরের ডেটায় সেটা ধরে থাকেনি (BRISC ৯৬.৫%,
Figshare ৯৮.২৭%) — তাই ১০০% মানে সব MRI-তে ১০০% নয়। এটা
**ডায়াগনোসিসের সহায়ক টুল**, ডাক্তারের বিকল্প নয়।
* টেস্ট সেট ছোট (২৪৬টি), আর ViT মডেলটা ভারী (৩২৮ MB, ৮৬M প্যারামিটার) —
CPU-তে চালাতে ধীর হতে পারে, GPU থাকলে ভালো চলে।
* মোবাইলে তোলা ছবি দিয়ে এখনো টেস্ট করা হয়নি; হাসপাতালের আসল ছবি
দিয়েও এখনো টেস্ট করা হয়নি — দুইটাই পরের ধাপ।

## ডেটাসেট (যাচাই করা)

* **Kaggle — Brain Tumor MRI Dataset** (masoudnickparvar):
[https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
* মোট **৭,০২৩টি** MRI ছবি, ৪ ক্লাস: Glioma ১,৬২১ · Meningioma ১,৬৪৫ ·
Pituitary Tumor ১,৭৫৭ · No Tumor ২,০০০। Training ৫,৭১২ / Testing ১,৩১১।
সাইজ ~১৫৬ MB, CC0 লাইসেন্স (ফ্রি ব্যবহার করা যায়)।
* সতর্কতা (রিপোর্টে লিখতে হবে): এই ডেটাসেটে কিছু প্রায়-একই ছবি থাকতে
পারে, তাই মাপা ৯৫.৯৩% অ্যাকুরেসি আসল হাসপাতালের ছবিতে কিছুটা কমতে
পারে — হাসপাতালের ডেটা দিয়ে টেস্ট করলেই আসল সংখ্যা জানা যাবে।
* **Hugging Face mirror (লগইন লাগে না, 2026-11-13 যাচাই করা):**
[https://huggingface.co/datasets/usamaJabar/brain-tumor-mri-classification-merged](https://huggingface.co/datasets/usamaJabar/brain-tumor-mri-classification-merged)
মোট **৯,০৪০টি** ছবি, সব 224×224 RGB: Train ৮,৩০৬ (glioma ২,১৮৩ ·
meningioma ১,৯১২ · no_tumor ২,০৪৪ · pituitary ২,১৬৭) · Val ৪৮৮ ·
Test ২৪৬। **ফাইনাল মডেলটা এই মিররের ডেটা দিয়েই ট্রেন করা হয়েছে**
(train.py-র Kaggle ডেটা নয়) — বিস্তারিত `DATASET_VERIFIED.md` দেখো।
`data/Training` আর `data/Testing` এই মিররের train/test ফোল্ডারের
symlink (no_tumor → notumor নামে), তাই `train.py` / `train_final.py`
কোনো বদল ছাড়াই চলে।
* **হাসপাতালের আসল ডেটা (ইফতি ও ত্বহা ম্যানেজ করবে):** রোগীর নাম-ঠিকানা ছাড়া
শুধু MRI ছবি (JPG/DICOM), ১০০–২০০টি, প্রতিটির সাথে ডাক্তারের
রিপোর্টের ক্লাস (কোন ছবিতে কোন রোগ — ফোল্ডারের নামে লেখা থাকবে)।
এটা দিয়ে শুধু **টেস্ট** করা হবে — Kaggle দিয়ে ট্রেন, হাসপাতালের
ছবি দিয়ে যাচাই। হাসপাতাল থেকে ছোট একটা অনুমতিপত্র নিয়ে রাখলে
রিপোর্টে লাগবে।

## ফোল্ডার কাঠামো (VS Code-এ যেমন খুলবে)

```
brain-tumor-project/
├── README.md               # এই ফাইল — চালানোর নিয়ম + মাপা ফলাফল
├── PROPOSAL.md             # পুরো প্রপোজাল (স্যারকে দেওয়ার মতো)
├── FINAL_STATUS.md         # FINAL-এ কী আছে, কী বাকি — সংক্ষেপে
├── DATASET_VERIFIED.md     # ডেটাসেট যাচাইয়ের বিবরণ
├── requirements.txt        # লাইব্রেরি (2026-11-13 যাচাই করা সংস্করণ)
├── download_dataset.py     # ডেটাসেট নামানো + data/ ফোল্ডারে সাজানো
├── model.py                # ViT (FINAL) + ResNet50 বেসলাইন + ResNet18 (তুলনার জন্য) + Grad-CAM
├── train.py                # ViT ট্রেনিং কোড (Google Colab GPU-তে চালানোর মতো)
├── train_final.py          # বেসলাইন ট্রেনিং কোড — ResNet18, CPU-তে চলে (তুলনার জন্য)
├── train_colab.ipynb       # Colab-এ GPU ট্রেনিংয়ের নোটবুক
├── predict.py              # একটা ছবি দিয়ে টেস্ট করার কোড
├── app.py                  # Flask ওয়েব অ্যাপ — আপলোড করলেই রেজাল্ট
├── templates/index.html    # ওয়েব পেজ (মোবাইল ক্যামেরা থেকেও ছবি তোলা যাবে)
├── static/style.css        # ওয়েব পেজের ডিজাইন
├── data/                   # ডেটাসেট (ZIP-এ থাকে না — download_dataset.py দিয়ে নামাতে হয়)
├── models/                 # best_model.pth (৩২৮ MB, ViT — Drive থেকে নিতে হয়) + metrics.json (ViT) + metrics-resnet18.json
├── train_final.log         # ট্রেনিংয়ের লগ (৪ epoch-এর হিসাব)
└── predict_test_results.txt # ৫টি ছবিতে predict.py পরীক্ষার ফল

```

## চালানোর নিয়ম — How to Run (Step by Step)

### ১. VS Code-এ খোলা

VS Code-এ **File → Open Folder** দিয়ে এই `brain-tumor-project` ফোল্ডারটা খোলো।

### ২. লাইব্রেরি ইনস্টল করা

```bash
python -m venv venv
venv\Scripts\activate        # Windows হলে; Mac/Linux হলে: source venv/bin/activate
pip install -r requirements.txt
# সতর্কতা: --system-site-packages দিয়ে venv বানাবে না — পুরনো system
# numpy/scipy/matplotlib-এর সাথে numpy 2.x-এর সংঘাত হয় (2026-11-13
# এই সমস্যাটা দেখা গিয়েছিল, পরিষ্কার venv-এ ঠিক হয়েছে)।
# CPU-only পিসিতে আগে চালাও:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

```

### ৩. ডেটাসেট নামানো (লগইন ছাড়াই হবে)

```bash
python download_dataset.py --mirror
#    — Hugging Face mirror থেকে ৯,০৪০ ছবি নামিয়ে data/ ফোল্ডারে সাজাবে
#    (Kaggle উপায় চাইলে: python download_dataset.py — kaggle.json লাগবে,
#     নিয়ম download_dataset.py-এর ওপরে লেখা আছে)

```

### ৪. ট্রেনিং — FINAL মডেলটা যেভাবে বানানো হয়েছে

```bash
# FINAL ViT মডেলটা Google Colab GPU-তে train_colab.ipynb দিয়ে বানানো হয়েছে:
# python train.py --epochs 15 --model vit --batch 32
# ফল: models/best_model.pth (৩২৮ MB) + models/metrics.json — Drive-এর brain-tumor-vit/ ফোল্ডারে সেভ আছে
# তুলনার জন্য ResNet18 বেসলাইন চাইলে: python train_final.py --epochs 4 --batch 32 --max-per-class 300

```

### ৫. একটা ছবি দিয়ে টেস্ট করা

```bash
python predict.py --image data/Testing/glioma/rf_test_glioma_725.jpg
python predict.py --image path/to/mri.jpg --heatmap result_heatmap.jpg
#    (ডিফল্ট মডেল vit — FINAL ট্রেইন করা মডেল; ResNet18 বেসলাইন চাইলে --model resnet18)

```

### ৬. ওয়েব অ্যাপ চালানো

```bash
python app.py
#    তারপর ব্রাউজারে খোলো: http://127.0.0.1:5000
#    MRI ছবি আপলোড করো অথবা মোবাইলের ক্যামেরা দিয়ে ছবি তোলো

```

## মোবাইল ক্যামেরা দিয়ে ছবি তুললে কি হবে?

হ্যাঁ, হবে — তবে ডিজিটাল ফাইল (JPG/DICOM) সরাসরি আপলোড করলে সবচেয়ে
ভালো ফল আসবে। মোবাইলে ফিল্ম/স্ক্রিনের ছবি তুললে অ্যাকুরেসি কিছুটা
কমতে পারে, তাই ট্রেনিংয়ে বাঁকা/ঝাপসা/আলো-কম ছবির augmentation রাখা
হয়েছে (`train.py`-তে `get_train_transforms` দেখো), আর অ্যাপে আপলোডের
আগে অটো-ক্রপ ও ছবি পরিষ্কার করার ধাপ আছে। ফাইনাল ডেমো ডিজিটাল ফাইল
দিয়ে দেখানোই ভালো; মোবাইলের ছবিটা এক্সট্রা ফিচার হিসেবে দেখাবে।

## সময়সূচি (৭–১০ দিন) — আসল অগ্রগতি

* দিন ১: প্রপোজাল + স্টার্টার কোড ✅
* দিন ২–৪: মডেল ট্রেনিং + টেস্ট ✅ — 2026-11-14 Colab GPU-তে ViT ট্রেন করা হয়েছে
(১৫ epoch, ~৭৩ মিনিট, টেস্ট অ্যাকুরেসি ১০০%; ResNet18 বেসলাইন ৯৫.৯৩%)
* দিন ৫–৬: ওয়েব অ্যাপ চূড়ান্ত করা — কোড রেডি, import + মডেল লোড
পরীক্ষা করা হয়েছে; ব্রাউজারে চালিয়ে দেখা বাকি
* দিন ৭–৮: ফাইনাল রিপোর্ট + প্রেজেন্টেশন স্লাইড — ⏳ বাকি
* দিন ৯–১০: হাসপাতালের আসল ছবি দিয়ে টেস্ট — ⏳ ইফতি ও ত্বহার ছবির অপেক্ষায়

## গুরুত্বপূর্ণ সত্য কথা

এটা **ডায়াগনোসিসের সহায়ক টুল**, ডাক্তারের বিকল্প নয় — রিপোর্টে এই
কথাটা স্পষ্ট লিখতে হবে। মডেল ভুল করতে পারে, তাই চূড়ান্ত সিদ্ধান্ত
সবসময় ডাক্তারের।