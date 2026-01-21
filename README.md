# Probabilistic 3D Tissue Motion Forecasting from Stereo Surgical Video

## Overview

This project implements probabilistic forecasting of 3D tissue motion in surgical video using deep learning approaches. The system predicts future disparity maps from stereo surgical video sequences, enabling anticipation of tissue deformation during minimally invasive surgery.

## Key Features

- **Multiple Architecture Support**: UNet and Swin-UNETR backbones
- **Probabilistic Forecasting**: Diffusion-based models for uncertainty quantification
- **Multi-Horizon Prediction**: Forecast at multiple time horizons (3, 5, 7, 9 frames)
- **Real-time Evaluation**: Streaming pipeline with latency analysis
- **Stereo Disparity Generation**: Multiple methods (RAFT-Stereo, SCARED toolkit)

## Dataset

The project uses the **SCARED (Stereo Correspondence and Reconstruction of Endoscopic Data)** dataset:

- Stereo surgical video sequences
- Multiple datasets: `dataset_1`, `dataset_2`, `dataset_3`
- Multiple keyframes per dataset: `keyframe_1` through `keyframe_5`
- Includes RGB video and disparity maps
- Data structure: `/scared_data/train/dataset_X/keyframe_Y/data/`

## Models and Approaches

### 1. Baseline Models

- **UNet**: Standard encoder-decoder architecture for disparity forecasting
- **Swin-UNETR**: Transformer-based model using Swin Transformer backbone with UNETR architecture

### 2. Diffusion-Based Models

Probabilistic forecasting using diffusion models:

- **Diffusion Head with UNet**: UNet backbone with diffusion head for uncertainty estimation
- **Diffusion Head with Swin-UNETR**: Transformer-based backbone with diffusion capabilities

### 3. Multi-Horizon Head

- Simultaneous prediction at multiple time horizons
- Horizons: 3, 5, 7, and 9 frames into the future
- Shared feature extraction with horizon-specific prediction heads

### 4. Disparity Generation

- **RAFT-Stereo**: State-of-the-art optical flow-based stereo matching
- **SCARED Toolkit**: Dataset-specific disparity generation tools

## Installation

### Prerequisites

```bash
# Python 3.8+
# CUDA-capable GPU (recommended)
```

### Dependencies

```bash
pip install torch torchvision
pip install monai[all] nibabel einops
pip install opencv-python
pip install imageio
pip install matplotlib pandas
pip install Pillow numpy
```

### Google Colab Setup

Most notebooks are designed to run on Google Colab with Google Drive mounted:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Usage

### 1. Disparity Generation

Start by generating disparity maps from stereo video:

```bash
# Using RAFT-Stereo
jupyter notebook raft_stereo_based_disparity_generation/generating_disparities_using_raft_stereo.ipynb

# Or using SCARED toolkit
jupyter notebook scared_toolkit_based_disparity_generation\ copy/generating_disparities_using_scared_toolkit.ipynb
```

### 2. Model Training

#### Train UNet Baseline

```bash
jupyter notebook UNet/Unet_Training.ipynb
```

#### Train Diffusion-based UNet

```bash
jupyter notebook UNet/Diffusion_Head_with_UNet-Training.ipynb
```

#### Train Swin-UNETR

```bash
jupyter notebook SwinUNETR/Swin_UNETR_Training.ipynb
```

#### Train Multi-Horizon Model

```bash
jupyter notebook Multi-Horizon-Head/MultiHorizonHead_Training.ipynb
```

### 3. Inference and Evaluation

#### Diffusion Sampling

```bash
jupyter notebook UNet/Diffusion_Head_with_UNet-Sampling.ipynb
```

#### Real-time Streaming Evaluation

```bash
jupyter notebook Final_pipeline/Streaming_Evaluation_with_Latency.ipynb
```

#### Frame Rate Analysis

```bash
jupyter notebook Final_pipeline/frame_rate_calculation.ipynb
```

## Pipeline Components

### Data Processing

- **Context Length**: 3 frames (past observations)
- **Forecast Horizons**: 3, 5, 7, 9 frames
- **Disparity Scaling**: Division by 256.0 for normalization
- **Image Cropping**: Random crops (256×320) for data augmentation

### Training Configuration

- **Datasets Used**: Configurable (typically dataset_1, dataset_2, dataset_3)
- **Keyframes**: Selectable subset (keyframe_1 through keyframe_5)
- **Device**: CUDA GPU (when available)
- **Data Split**: Train/validation split with random sampling

### Evaluation Metrics

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Latency analysis for real-time performance
- Visual quality assessment through disparity visualization

Key achievements:

- Multi-horizon forecasting at various time scales
- Probabilistic predictions with uncertainty estimation
- Real-time capable inference with latency analysis
- Comparison of deterministic vs. probabilistic approaches

## Model Checkpoints

Trained model checkpoints are saved in the `output/` directory with naming conventions:

- UNet models: `unet_*.pth`
- Swin-UNETR models: `swin_unetr_*.pth`
- Multi-horizon models: `multi_horizon_*.pth`
- Diffusion models: `diffusion_*.pth`

## Configuration

### Data Paths

Update the base directory path in each notebook:

```python
BASE_DIR = Path("/content/drive/Shareddrives/TissueMotionForecasting")
TRAIN_ROOT = BASE_DIR / "scared_data" / "train"
```

### Hyperparameters

Key hyperparameters that can be adjusted:

- `CONTEXT_LEN`: Number of past frames for context (default: 3)
- `FORECAST_HORIZON`: Prediction horizon(s)
- `DISP_SCALE`: Disparity normalization factor (default: 256.0)
- `BATCH_SIZE`: Training batch size
- `LEARNING_RATE`: Optimizer learning rate
- `NUM_EPOCHS`: Training duration

## Visualization

The notebooks include visualization utilities for:

- Input disparity sequences
- Predicted future disparity maps
- Ground truth comparisons
- Colored disparity visualizations
- Error heatmaps

## Hardware Requirements

### Minimum

- GPU: 8GB VRAM (e.g., NVIDIA RTX 2070)
- RAM: 16GB
- Storage: 150GB for dataset and checkpoints

### Recommended

- GPU: 16GB+ VRAM (e.g., NVIDIA V100, A100)
- RAM: 32GB+
- Storage: 150GB+

## Future Work

Potential extensions and improvements:

- Temporal attention mechanisms
- Online learning and adaptation
- Integration with surgical navigation systems
- Multi-modal fusion (RGB + depth + flow)
- Improved uncertainty quantification
- Extended evaluation on additional surgical datasets
