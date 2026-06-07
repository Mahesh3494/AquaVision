# 🦐 AquaVision

ML-powered shrimp disease detection tool for aquaculture farmers.

## What It Does
Farmer uploads a photo of their shrimp → app returns:
- Disease identification
- Confidence score
- Severity level
- Recommended action

## Diseases Detected
| Disease | Description |
|---|---|
| Healthy | No disease detected |
| BG | Black Gill Disease |
| WSSV | White Spot Syndrome Virus |
| WSSV_BG | Combined WSSV + Black Gill infection |

## Model
- Architecture: EfficientNetB0 (Transfer Learning)
- Dataset: 2000 images, 4 classes, 500 per class
- Validation Accuracy: 85%

## Tech Stack
- Model: TensorFlow / Keras / EfficientNetB0
- App: Streamlit
- Training: Google Colab (T4 GPU)

## Try It Live
[Link coming soon]

## Built By
Mahesh Penubothu — VIT-AP University CSE