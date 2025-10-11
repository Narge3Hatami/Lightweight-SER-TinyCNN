# Lightweight Speech Emotion Recognition (SER)

This repository contains the core implementation for the paper: **"Lightweight Speech Emotion Recognition Using Open-Source Tools and TinyCNN on Low-Resource Devices"**.

This project provides a lightweight and reproducible framework for SER, optimized for deployment on standard CPU hardware.

## Quick Start

### 1. Installation
Clone the repository and install the required Python libraries.

### 2. Dataset Setup
Download the RAVDESS, TESS, and CREMA-D datasets and place them in a root `datasets` folder.

### 3. Preprocessing
Run a preprocessing script (using the functions in `preprocessing.py`) to generate the `.npy` files for spectrograms and features. This script should create a `processed_data` directory.

### 4. Training and Evaluation
To run the evaluation pipeline for the TinyCNN model on a specific dataset, execute the main script from your terminal:
```bash
python train_evaluate.py --dataset RAVDESS
```
You can replace `RAVDESS` with `TESS` or `CREMA-D`.

## Core Components
- **`preprocessing.py`**: Contains functions for audio preprocessing and feature/spectrogram extraction.
- **`model.py`**: Defines the TinyCNN model architecture.
- **`train_evaluate.py`**: The main script to reproduce the paper's results for the TinyCNN model.

## Citation
If you find this work useful in your research, please consider citing our paper. The BibTeX entry is provided below. Please update the `journal` and `year` fields once the paper is available on arXiv.

```bibtex
@article{TinyCNN_SER2025,
  title   = {Lightweight Speech Emotion Recognition Using Open-Source Tools and TinyCNN on Low-Resource Devices},
  author  = {Ali Jafari and Narges Hatami},
  journal = {arXiv preprint (to appear)},
  year    = {2024},
  note    = {Code: \url{https://github.com/Narge3Hatami/Lightweight-SER-TinyCNN}}
}
