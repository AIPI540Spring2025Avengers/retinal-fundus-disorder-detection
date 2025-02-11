
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

## Setup

```bash
./setup.sh
```

This script takes care of setting up your virtual environment if it does not already exist, activating it, installing requirements,
pulling the dataset (if not already present in the data directory), and then running the traditional and deep learning modeling scripts.
At any point that a matplotlib graph is opened, please close it to continue the script execution.

## Running the Streamlit application locally

Assuming your virtual environment is setup and activated, and that the requirements are installed from running the previous script,
you can then run the following to startup a local instance of the Streamlit application.

```bash
python main.py
```

## Dataset & License
This repository uses the [Retinal Fundus Images](https://www.kaggle.com/datasets/kssanjaynithish03/retinal-fundus-images) dataset from Kaggle, licensed under CC-BY-NC-SA 4.0.

## Canva Presentation

View our Canva presentation [HERE](https://www.canva.com/design/DAGecblSyBM/Y-as2fyST6t9cdKNydDN_Q/view?utm_content=DAGecblSyBM&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=hb1292d3d79).
