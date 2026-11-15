# FINAL STATUS — Brain Tumor Detection Project (2026-11-14, ViT FINAL)

## ✅ FINAL — যা সম্পূর্ণ হয়েছে
- **FINAL মডেল ViT:** Vision Transformer (`vit_base_patch16_224`, timm),
  Colab T4 GPU-তে ট্রেন করা — ১৫ epoch, পুরো ৮,৩০৬ ছবি, 224×224,
  সেরা epoch ৫, সময় ~৭৩ মিনিট।
- **আসল মাপা ফল:** টেস্ট অ্যাকুরেসি **১০০%** (২৪৬ ছবিতে ২৪৬ সঠিক),
  Macro F1 ১.০। পুরো হিসাব `models/metrics.json`-এ।
  - Glioma ৮১/৮১, Meningioma ৫৫/৫৫, No Tumor ৪৯/৪৯, Pituitary ৬১/৬১
- **বাইরের টেস্ট (একই ViT, নতুন করে ট্রেন নয়):**
  BRISC ২০২৫ — ৯৬.৫% (৯৬৫/১,০০০), Figshare Cheng — ৯৮.২৭% (৩,০১১/৩,০৬৪)।
  গড়: শতাংশের গড় ৯৮.২৬%, ছবি-ওজন করা গড় ৯৭.৯৬% (৪,২২২/৪,৩১০)।
- **তুলনার জন্য বেসলাইন ResNet18:** ৯৫.৯৩% (২৪৬-এ ২৩৬ সঠিক),
  Macro F1 ০.৯৫৭ — হিসাব `models/metrics-resnet18.json`-এ,
  মডেল `models/best_model_resnet18.pth` (৪৩ MB)।
- **কোড ViT-এ বদলানো হয়েছে:** `app.py` আর `predict.py` এখন ViT লোড করে
  (ডিফল্ট `--model vit`)। UI-তে Vision Transformer + ১০০% Accuracy দেখায়।
- **ডেটাসেট যাচাই করা:** Hugging Face mirror, ৯,০৪০ ছবি
  (train ৮,৩০৬ / val ৪৮৮ / test ২৪৬), সব 224×224 RGB
  (`DATASET_VERIFIED.md`)।

## ⚠️ মডেল ফাইল — ZIP-এ নেই, Drive থেকে নিতে হবে
- ViT-এর `best_model.pth` (৩২৮ MB) শুধু Google Drive-এ আছে:
  `brain-tumor-vit` ফোল্ডার (chatgp3004@gmail.com),
  md5 `b6dbc0ec6cf86a2789011867ff43b874`।
- ZIP-এ `models/best_model.pth` নেই — ডাউনলোড করে `models/` ফোল্ডারে
  `best_model.pth` নামে রাখলেই `python app.py` / `predict.py` চলবে।
  নিয়ম: `models/GET_VIT_MODEL.md` দেখো।
- কারণ: Drive কানেক্ট করা নেই বলে এই VM থেকে ফাইলটা নামানো যায়নি,
  আর ৩২৮ MB ফাইল ZIP-এ ঢোকালে ZIP-ও ৩৩০ MB+ হয়ে যেত।
- ভুল এড়াতে ResNet18-এর ফাইলটার নাম বদলে
  `best_model_resnet18.pth` করা হয়েছে — যাতে ViT-এর নামে ResNet
  মডেল ভুল করে চালানো না হয়।

## ⏳ এখনো বাকি (FINAL-এর বাইরে)
1. **হাসপাতালের আসল ছবি দিয়ে টেস্ট:** ইফতি হাসপাতাল থেকে anonymised
   MRI ছবি ম্যানেজ করবে বলেছে — ছবি পেলে `predict.py` দিয়ে টেস্ট করে
   ফল যোগ করতে হবে।
2. **মোবাইলে তোলা ছবি দিয়ে টেস্ট:** এখনো করা হয়নি।
3. **ফাইনাল রিপোর্ট / থিসিস + প্রেজেন্টেশন স্লাইড:** খসড়া আছে,
   ViT-এর ১০০% + বাইরের টেস্টের সংখ্যা দিয়ে আপডেট করতে হবে।
4. **PDF রিপোর্ট বোতাম:** `app.py`-র `/report` এখনো placeholder।

## ⚠️ সৎ সীমাবদ্ধতা
- ১০০% সংখ্যাটা ২৪৬টি টেস্ট ছবির ওপর মাপা — বাইরের ডেটায় সেটা
  ৯৬.৫% / ৯৮.২৭%-এ নেমেছে। BRISC-এ সুস্থ (No Tumor) ১৪০টার মধ্যে
  ২৩টাকে ভুল করে টিউমার বলেছে — এটা রিপোর্টে লিমিটেশন হিসেবে লিখতে হবে।
- এটা **ডায়াগনোসিসের সহায়ক টুল**, ডাক্তারের বিকল্প নয়।
- ViT মডেলটা ভারী (৩২৮ MB, ৮৬M প্যারামিটার) — CPU-তে চালাতে ধীর হতে পারে।
