# Brain Tumor MRI Dataset — verified on this machine (2026-11-13)

## Primary (Kaggle, as in PROPOSAL.md)
- Brain Tumor MRI Dataset, masoudnickparvar
- https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- 7,023 images, 4 classes: glioma 1,621 / meningioma 1,645 / pituitary 1,757 / no tumor 2,000
- Train 5,712 / Test 1,311, ~156 MB, CC0 licence
- Needs a free Kaggle account + kaggle.json (see download_dataset.py)

## Mirror verified here — no login needed (Hugging Face)
- https://huggingface.co/datasets/usamaJabar/brain-tumor-mri-classification-merged
- File downloaded and verified on 2026-11-13:
  `data/brain-tumor-mri-classification.zip` — 236 MB
- Unzipped to `data/merged_dataset_v3/images/` — **9,040 images total**, all 224x224 RGB JPG:
  - train: glioma 2,183 / meningioma 1,912 / no_tumor 2,044 / pituitary 2,167 = 8,306
  - val: glioma 161 / meningioma 109 / no_tumor 96 / pituitary 122 = 488
  - test: glioma 81 / meningioma 55 / no_tumor 49 / pituitary 61 = 246
- `data/Training/` and `data/Testing/` are symlinks to the merged train/test
  folders (with `no_tumor` linked as `notumor`, to match the class order in
  model.py: glioma, meningioma, notumor, pituitary), so `train.py` works
  unchanged with this mirror.
- The 236 MB zip and the extracted images are NOT included in the delivery
  ZIP (too large). Ifti downloads it with `python download_dataset.py --mirror`
  or from the Hugging Face link above.

## Still needed
- Hospital real MRI images (anonymised, labelled by doctor's report) for
  Phase 2 testing — Ifti said he can manage these from a hospital.
- Training itself needs a GPU (Google Colab, free T4) — this VM has no
  torch/GPU, so `train.py` was syntax-checked only, not run end-to-end here.
  Use `train_colab.ipynb` in Colab.
