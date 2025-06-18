# Detecting-AI-and-real-images

## Demo
Check out the live demo at: https://detecting-ai-images.streamlit.app/  
![me](https://github.com/songthienll/Detecting-AI-and-real-images/blob/main/demo.gif)

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
   - Available on Hugging Face: https://huggingface.co/songthienll/mobilevit_ai_real_classifier
2. **Swin-T**:  
   - Swin-Tiny (a variant of the Swin Transformer) adopts a hierarchical architecture and multi-resolution feature processing, making it a highly versatile and powerful model for various computer vision tasks. It uses a sliding window-based attention mechanism that allows for efficient scaling on large image sizes.
   - Parameters: ~28 million.  
   - Available on Hugging Face: https://huggingface.co/songthienll/swint-model

3. **ConvNeXt-T**:  
   - ConvNeXt-Tiny is a modernized Convolutional Neural Network (CNN) inspired by the advancements in Vision Transformers. It provides state-of-the-art performance while retaining the simplicity and efficiency of traditional CNNs.
   - Parameters: ~29 million.  
   - Available on Hugging Face: https://huggingface.co/songthienll/convnext-t-model

These models were fine-tuned on a dataset of real and AI-generated images, with inputs normalized to 224x224 pixels.

## Dataset

The dataset consists of real images and AI-generated images (e.g., from diffusion models). Due to its large size, it is not included in the repository. You can download my dataset from [this link](https://www.kaggle.com/datasets/songthien/ai-and-real-images) or create a similar dataset yourself.
