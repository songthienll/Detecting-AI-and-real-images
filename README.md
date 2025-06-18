# Detecting-AI-and-real-images

## Introduction

This project focuses on detecting AI-generated images versus real images using advanced deep learning models such as Mobile-ViT-S, Swin-T, and ConvNeXt-T. The demo application is built with Streamlit, allowing users to upload images or videos to check their authenticity based on trained models.

## Installation

To run the project, you need the following libraries:

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- Streamlit
- OpenCV
- NumPy
- Pandas

Install the required libraries with the following command:

```bash
pip install -r requirements.txt
```


## Usage

To run the demo application, use the following command in the terminal:

```bash
streamlit run app.py
```

After running, access `http://localhost:8501` in your browser to use the interface. You can upload images or videos for analysis.

## Model Information

The project utilizes three main models, each optimized for high-performance computer vision tasks, offering a balance of efficiency and accuracy:

1. **Mobile-ViT-S**:  
   - Mobile-ViT-S is a lightweight, general-purpose Vision Transformer designed by merging the strengths of Convolutional Neural Networks (CNNs) and Transformers. Its architecture excels in balancing computational efficiency and high accuracy, making it highly suitable for resource-constrained environments such as mobile and edge devices.
   - Parameters: ~5.6 million.  

2. **Swin-T**:  
   - Swin-Tiny (a variant of the Swin Transformer) adopts a hierarchical architecture and multi-resolution feature processing, making it a highly versatile and powerful model for various computer vision tasks. It uses a sliding window-based attention mechanism that allows for efficient scaling on large image sizes.
   - Parameters: ~28 million.  


3. **ConvNeXt-T**:  
   - ConvNeXt-Tiny is a modernized Convolutional Neural Network (CNN) inspired by the advancements in Vision Transformers. It provides state-of-the-art performance while retaining the simplicity and efficiency of traditional CNNs.
   - Parameters: ~29 million.  


These models were fine-tuned on a dataset of real and AI-generated images, with inputs normalized to 224x224 pixels.

## Dataset

The dataset consists of real images and AI-generated images (e.g., from diffusion models). Due to its large size, it is not included in the repository. You can download a sample dataset from [this link](https://www.kaggle.com/datasets/songthien/ai-and-real-images) or create a similar dataset yourself.

## Demo

The demo application (via `app.py`) offers the following features:  
- **Image Upload**: Checks if an image is real or AI-generated, with a confidence score.  
- **Video Upload**: Analyzes individual frames (e.g., 1440 frames at 30.0 FPS, 1520x1680 resolution) and displays results (real/fake) for sample frames.  
- **Download Results**: Outputs a video with labeled predictions.
