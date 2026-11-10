# AquaVision retraining: handoff notes

Written 2026-11-10. Retraining is **parked** until next semester (deep learning course project).
Paste this whole file into a new chat when you start retraining.

## Why retrain
The current models were trained on poor images. Known failures (from AquaManage `docs/progress.md`):
- A clear WSSV photo came back **Healthy at 92.7%** (confident and wrong).
- A banner on the photo changes the answer; a tight crop flips Black gill to Healthy.

**Likely cause (not proven):** the shrimp model was trained on `data/Augmented_shrimp/`: exactly
500 photos per class, made by rotating/flipping fewer originals. If copies of the same photo end up
in both training and validation, the validation score goes up but nothing new was learned. That
would explain an 85% validation score next to bad results on real photos. **First check when
retraining:** split train/validation by *original photo*, never by augmented copy.

**Found 2026-11-10, testing on a phone:** a screenshot of the AquaManage login page (no shrimp
at all) came back **Black gill, 99%**. The model has only 4 classes, so it must pick one of them
for *any* picture, often very confidently. Margin can't catch this either. When retraining, add a
way to say "this isn't a shrimp photo": a 5th class trained on non-shrimp pictures, or a separate
out-of-distribution check before the disease model runs.

## What you have on this laptop (`C:\aquavision\data\`, gitignored, not on GitHub)
| Folder | Contents |
|---|---|
| `Augmented_shrimp/{BG,Healthy,WSSV,WSSV_BG}` | 500 each, augmented (current shrimp training set) |
| `ShrimpImages/ShrimpImages` | 1,149 images, not sorted into classes |
| `Fish Disease Dataset/train/<8 classes>` | 250 each (+ a `test/` folder); Kaggle "South Asia" fish set |

Model files: `shrimp_model.onnx` (EfficientNet-B0, 224 px, 4 classes, live),
`fish_model_b3.onnx` (EfficientNet-B3, 300 px, 8 classes, live), `.h5` Keras originals,
`train.py`, `convert_to_onnx.py`.

## What the dataset-search chat did (2026-11-05, cloud session, chat is lost)
It searched GitHub for extra images (Kaggle, Mendeley and Roboflow were blocked there), sorted
them into your class folder names, removed exact duplicates (same MD5 hash = byte-identical
files), skipped augmented copies, and pushed them to a branch.

**Where:** GitHub `Mahesh3494/AquaVision`, branch **`claude/amazing-thompson-horlme`**
(commit `e202010`), folder `datasets/`. 3,472 images, ~116 MB. Verified 2026-11-10: the branch
exists and the counts below match.

Download as zip:
`https://github.com/Mahesh3494/AquaVision/archive/refs/heads/claude/amazing-thompson-horlme.zip`

**Do not merge this branch into `main`**: 116 MB of images in the repo Streamlit Cloud deploys.
Download it, use the images, and later move them somewhere outside git.

| Folder on the branch | Images | Worth anything? |
|---|---|---|
| `fish/Bacterial Red disease` | 622 | partly new ("Red spot" from Muruli-KR, mapped to this name) |
| `fish/Bacterial gill disease` | 599 | partly new ("Gill rot" from Muruli-KR, mapped) |
| `fish/EUS Disease` | 423 | probably new (Muruli-KR) |
| `fish/Healthy Fish` | 674 | partly new |
| `fish/Bacterial diseases - Aeromoniasis` | 250 | probably your old Kaggle images again |
| `fish/Fungal diseases Saprolegniasis` | 250 | probably old |
| `fish/Parasitic diseases` | 249 | probably old |
| `fish/Viral diseases White tail disease` | 250 | probably old |
| `shrimp/Healthy` | 117 | new |
| `shrimp/WSSV` | 38 | new |
| `shrimp/BG`, `shrimp/WSSV_BG` | 0 | nothing found |

Sources (from the commit message): `Muruli-KR/fish_disease_dataset`,
`csolis97/Freshwater-Fish-Disease-Aquaculture`,
`Dimuth94/Freshwater-fish-disease-Identification-System` (originals only),
`eshaanmathakari/shrimpInfectionDetection`, `lorenzo-querol/wssv-recognition`.

**Not done:** nobody looked at the images by eye, checked the label mapping ("Gill rot" really
= bacterial gill disease?), or compared them with your existing folders for near-duplicates
(only byte-identical files were removed). The 250-per-class fish folders are very likely
copies of what you already have.

**Bottom line for shrimp:** the branch adds only 155 shrimp photos and none for BG or WSSV_BG.
It doesn't fix the shrimp model. The datasets below are the real fix.

## Better datasets to download yourself (on Mendeley Data; search the name there)
Numbers are what the old chat reported; check them on the dataset page.
1. **ShrimpDiseaseImageBD**: exactly your 4 classes from real farms. Healthy 403, BG 198,
   WSSV 328, BG_WSSV 220 (= your `WSSV_BG`). **Start here: it's the main fix for shrimp.**
2. **TigerShrimpBD**: Black gill, WSSV, Healthy. 3,574 images, but only the **1,001
   originals** are worth using; the rest are augmented. (Tiger shrimp = monodon.)
3. **MatsyaDx-BD** (fish): 2,137 farm photos of bacterial gill, bacterial red, EUS and healthy,
   across 3 carp species.

Check each dataset's licence before putting any of its photos on a public page.

## Plan when you retrain
1. Download ShrimpDiseaseImageBD (+ TigerShrimpBD originals). Look through a sample by eye.
2. Merge with your originals (not `Augmented_shrimp`), remove duplicates, keep a list of where
   every image came from.
3. Split train / validation / test **by original photo**, before any augmentation; augment only
   the training split, at training time.
4. Retrain, then test on photos the model never saw, including real farm photos (still to get
   from the farm) and the photos that fail now (the clear WSSV one).
5. Export to ONNX. **Keep the same input size, class order and output (softmax over the 4
   classes)**, or update the AquaVision API and AquaManage together. AquaManage only sees the
   API, so a better model with the same contract drops in with no AquaManage change.
6. Fish is lower priority: AquaManage v1 is shrimp only.

## Meanwhile (now)
AquaManage integrates the **current** model as it is. The result screen always says it is a
check, not a diagnosis.
