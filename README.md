# Retinal Fundus Disorder Detection

## Overview
This project implements a **retinal fundus image classification system** using three approaches; **naive approach**, **traditional machine learning** and **deep learning models**. The goal is to analyze and classify retinal images to identify potential disorders such as **Diabetic Retinopathy, Glaucoma, AMD, Cataracts, etc.**.

## Approaches 

### Naïve Rule-Based Classifier
- Uses Edge Density, Brightness, and Optic Disc detection for rule-based classification.

### Traditional Machine Learning (SVM Model)
- Uses Local Binary Pattern, Color Histograms (LAB space), and Vessel Magnitude as features; using SVM classifier with PCA for dimensionality reduction.

### Deep Learning (EfficientNet-B4 CNN)
- A fine-tuned EfficientNet-B4 model with custom classifier layers.


## User Interface

- **Streamlit Web App**: UI to classify uploaded images and display prediction probabilities for all three approaches.
