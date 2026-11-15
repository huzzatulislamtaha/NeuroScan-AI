# Project Proposal

## Project Title
**AI-Powered Brain Tumor Detection and Classification from MRI using Vision Transformer**

বাংলা: ভিশন ট্রান্সফরমার দিয়ে MRI থেকে ব্রেইন টিউমার শনাক্তকরণ ও শ্রেণিবিভাগ

**Student:** Abdullah Mahfuz Ifty (আব্দুল্লাহ মাহফুজ ইফতি)
**ID:** CSE2401031092, Section 31B
**Department:** B.Sc. in Computer Science and Engineering, Sonargaon University, Dhaka

## 1. Problem Statement (বাংলাদেশের সমস্যা)
বাংলাদেশে ব্রেইন টিউমার প্রায়ই ধরা পড়ে অনেক দেরিতে। ঢাকার বাইরে
অনেক হাসপাতালে MRI মেশিন থাকলেও রিপোর্ট পড়ার মতো বিশেষজ্ঞ
রেডিওলজিস্ট সব জায়গায় নেই, আর রিপোর্ট পেতে ৩–৪ দিন লেগে যায়।
দেরিতে ধরা পড়লে চিকিৎসা কঠিন ও ব্যয়বহুল হয়ে যায়। একটি AI টুল
MRI ছবি দেখে কয়েক সেকেন্ডে প্রাথমিক ধারণা দিতে পারলে ডাক্তারদের
কাজ অনেক দ্রুত হবে, বিশেষ করে গ্রামের হাসপাতালে।

## 2. Objectives (উদ্দেশ্য)
1. MRI ছবি থেকে ব্রেইন টিউমার আছে কি না শনাক্ত করা।
2. টিউমার থাকলে তার ধরন চেনা — Glioma, Meningioma, Pituitary Tumor।
3. প্রতিটি prediction-এর সাথে confidence % দেখানো।
4. Grad-CAM heatmap দিয়ে ছবিতে দেখানো — টিউমারটা কোথায় আছে।
5. মোবাইল ক্যামেরায় তোলা MRI ছবিতেও কাজ করে, এমন শক্ত মডেল বানানো।
6. একটি সহজ ওয়েব অ্যাপ বানানো, যেখানে MRI আপলোড করলেই রেজাল্ট আসে।

## 3. Dataset (ডেটাসেট)
- **প্রধান ডেটাসেট:** Kaggle — Brain Tumor MRI Dataset (masoudnickparvar)
  https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- মোট **৭,০২৩টি** MRI ছবি, ৪ ক্লাস:
  Glioma ১,৬২১ · Meningioma ১,৬৪৫ · Pituitary Tumor ১,৭৫৭ · No Tumor ২,০০০।
  Training ৫,৭১২ / Testing ১,৩১১। সাইজ ~১৫৬ MB, CC0 লাইসেন্স।
- **যাচাইয়ের ডেটা (হাসপাতাল থেকে):** রোগীর নাম-ঠিকানা ছাড়া শুধু
  MRI ছবি, ১০০–২০০টি, প্রতিটির সাথে ডাক্তারের রিপোর্টের ক্লাস।
  Kaggle দিয়ে ট্রেন, হাসপাতালের ছবি দিয়ে টেস্ট করা হবে।
- সতর্কতা: Kaggle ডেটাসেটে কিছু প্রায়-একই ছবি থাকতে পারে, তাই
  রিপোর্টে split-এর সীমাবদ্ধতা সৎভাবে লেখা হবে।

## 4. Methodology (কীভাবে করবো)
1. **Data collection:** Kaggle থেকে ডেটাসেট নামানো (`download_dataset.py`)।
2. **Preprocessing:** সব ছবি 224×224 করা, normalize করা।
3. **Augmentation:** rotation, perspective, brightness, blur —
   মোবাইলে তোলা ছবির জন্য মডেলকে শক্ত করা।
4. **Model:** মূল মডেল Vision Transformer (`vit_base_patch16_224`,
   ImageNet pretrained), বেসলাইন হিসেবে ResNet50 — একই ডেটায়
   দুই মডেলের তুলনা করা হবে।
5. **Training:** Cross-entropy loss, AdamW optimizer, ১৫ epoch,
   Google Colab GPU-তে।
6. **Evaluation:** accuracy, precision, recall, F1-score,
   confusion matrix।
7. **Explainability:** Grad-CAM heatmap (ResNet50-এ আসল Grad-CAM,
   ViT-এ gradient saliency)।
8. **Web app:** Flask অ্যাপ — আপলোড / মোবাইল ক্যামেরা → রেজাল্ট।

## 5. Tech Stack
- **Language:** Python 3.10+
- **Deep learning:** PyTorch, torchvision, timm (Vision Transformer)
- **Data:** NumPy, Pillow, OpenCV, scikit-learn, Matplotlib
- **Web:** Flask, HTML/CSS
- **Training:** Google Colab (GPU)

## 6. Timeline (৭–১০ দিন)
| দিন | কাজ |
|---|---|
| ১ | প্রপোজাল + স্টার্টার কোড (Phase 1) ✅ |
| ২–৪ | মডেল ট্রেনিং + টেস্ট (Colab GPU) |
| ৫–৬ | ওয়েব অ্যাপ চূড়ান্ত করা |
| ৭–৮ | ফাইনাল রিপোর্ট + প্রেজেন্টেশন স্লাইড |
| ৯–১০ | টেস্টিং + ফিডব্যাক অনুযায়ী ঠিক করা |

## 7. Expected Results (আশা করা ফল)
- **Test accuracy: ৯৫%+ (লক্ষ্য — এখনো মাপা ফল নয়, ট্রেনিংয়ের পর
  আসল সংখ্যা বসবে।)**
- ৪ ক্লাসের প্রতিটির precision / recall / F1-score রিপোর্টে থাকবে।
- Grad-CAM heatmap-সহ লাইভ ডেমো — MRI আপলোড করলেই রেজাল্ট।
- হাসপাতালের আসল ছবি দিয়ে যাচাই — মডেল শুধু Kaggle ডেটায়
  নয়, বাংলাদেশের হাসপাতালের ছবিতেও কাজ করে কি না।

## 8. Limitations (সীমাবদ্ধতা — রিপোর্টে লিখতে হবে)
- এটা **ডায়াগনোসিসের সহায়ক টুল, ডাক্তারের বিকল্প নয়**।
- মডেল ভুল করতে পারে; চূড়ান্ত সিদ্ধান্ত সবসময় ডাক্তারের।
- মোবাইলে তোলা ছবিতে অ্যাকুরেসি ডিজিটাল ফাইলের চেয়ে কম হতে পারে।
- রোগীর ব্যক্তিগত তথ্য ছাড়া, শুধু anonymized ছবি ব্যবহার করা হবে।
