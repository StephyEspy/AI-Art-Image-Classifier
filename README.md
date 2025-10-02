# AI Art Image Classifier

CNN (ResNet-50, transfer learning) to classify artwork as AI-generated vs. human-created.

## Overview
- Task: Binary image classification (AI vs. human art)
- Data: Combined public datasets (~180k images)
- Approach: Transfer learning on ResNet-50 + on-the-fly augmentation
- Result: ~88–90% validation accuracy (binary)

## How to Run
- Open the notebook in Google Colab (link here) or run locally with TensorFlow/Keras.
- Expected input size: 224x224 RGB
- Basic steps: install deps → place dataset folders → run training cell

## Repo Notes
- Originally built with a team of 8; this repo is my consolidated version.
- My focus areas: [e.g., augmentation pipeline, fine-tuning schedule, evaluation/metrics].

## Files
- `src/` – training + model code
- `figures/` – plots/images
- `outputs/` – logs/checkpoints
- `project-proposal.md` – initial project plan

## Next Steps
- Add Grad-CAM visualizations
- Compare EfficientNet/MobileNet backbones
- Export weights + simple demo app (Streamlit)
