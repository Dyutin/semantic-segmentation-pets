# Semantic Segmentation on Pet Images Using U-Net

This repository contains code and experiments for a computer vision mini-project on semantic segmentation using U-Net. The project focuses on distinguishing cats, dogs, and background from the Oxford-IIIT Pet Dataset, and includes analysis of robustness under various image perturbations.

## 📌 Overview

We implemented and compared two models:
- A **standard U-Net** trained end-to-end.
- A **U-Net with encoder pretrained** via an autoencoder.

## 🔍 Results

| Model                   | Class      | IoU (%) | Dice (%) |
|-------------------------|------------|---------|----------|
| **Standard U-Net**      | Background | 92.28   | 95.99    |
|                         | Cat        | 66.79   | 80.09    |
|                         | Dog        | 71.51   | 83.39    |
|                         | **Mean**   | **76.86** | **86.49** |
| **Autoencoder U-Net**   | Background | 85.75   | 92.33    |
|                         | Cat        | 35.79   | 52.72    |
|                         | Dog        | 49.43   | 66.15    |
|                         | **Mean**   | **56.99** | **70.40** |

> **Table:** Test set performance (outline class excluded from metrics).

## Project Structure

Evaluate/ # Robustness evaluation scripts
autoencoder.py # Autoencoder model definition
inference.py # Inference utilities
pet_dataset.py # Data loading & augmentations
sweep.py # Hyperparameter sweep
train_autoencoder.py # Pretrain encoder
train_frozen_unet.py # Train U-Net with frozen encoder
train_unet.py # Train standard U-Net
unet.py, unet_parts.py # U-Net modules



## Model Architecture

- **Input size:** 256×256  
- **Classes:** Background (0), Cat (1), Dog (2), Outline (3)  
- **Loss:** Weighted Cross-Entropy  
- **Metrics:** Intersection over Union (IoU), Dice Coefficient  

## Robustness Experiments

Tested under:
- Gaussian noise  
- Blurring  
- Brightness & contrast shifts  
- Occlusions  
- Salt-and-pepper noise  

