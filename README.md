# ♻️ Garbage Classification using CNN

A Computer Vision project that classifies waste images into five categories using a **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras** and deployed using **Streamlit**.

---

## 📌 Project Overview

Waste classification plays an important role in improving recycling efficiency and reducing environmental impact. This project develops a CNN-based image classification model capable of recognizing several common recyclable waste categories.

The application provides:

- 🏠 Home page
- 📊 Exploratory Data Analysis (EDA)
- 🤖 Image Prediction

Users can upload their own waste images and receive the predicted class along with the confidence score and probability distribution for each class.

---

## 📂 Dataset

This project uses the **Garbage Classification Dataset** from Kaggle:

**Dataset Source**

https://www.kaggle.com/datasets/mostafaabla/garbage-classification

The original dataset contains **12 waste categories**:

- Battery
- Biological
- Brown Glass
- Cardboard
- Clothes
- Green Glass
- Metal
- Paper
- Plastic
- Shoes
- Trash
- White Glass

For this project, only **5 classes** were selected:

- 📄 Paper
- 📦 Cardboard
- 🧴 Plastic
- 🥫 Metal
- 🍾 White Glass

The selected classes were used throughout the training, evaluation, and deployment stages.

---

## 🤖 Model

- Framework : TensorFlow / Keras
- Architecture : Convolutional Neural Network (CNN)
- Input Size : **224 × 224**
- Number of Classes : **5**

The deployment uses the same preprocessing pipeline and trained model as the training and inference notebooks.

---

## 📊 Features

### 🏠 Home

- Project overview
- Dataset information
- Workflow
- Waste categories
- Application usage guide

### 📊 Exploratory Data Analysis (EDA)

- Dataset summary
- Class distribution
- Class percentage
- Number of images per class
- Sample images
- Waste characteristics
- Image resolution
- Preprocessing summary

### 🤖 Prediction

Users can upload their own image to:

- Preview uploaded image
- View file information
- Predict waste category
- Display confidence score
- View prediction probability table
- Visualize prediction probabilities

---

## 📁 Project Structure

```text
Garbage-Classification-CNN/
│
├── app.py
├── home.py
├── eda.py
├── prediction.py
│
├── garbage_classification_mobilenetv2.keras
│
├── garbage_dataset_5class/
│   ├── cardboard/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── white-glass/
│
├── requirements.txt
├── README.md
│
└── .streamlit/
    └── config.toml
```

---

## 🚀 Installation

Clone this repository:

```bash
git clone https://github.com/USERNAME/REPOSITORY_NAME.git
```

Move into the project directory:

```bash
cd REPOSITORY_NAME
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 🌐 Streamlit Deployment

Deployment Link:

> _Coming Soon_

---

## 👨‍💻 Author

**Khalfani Novian Habibi**

Computer Vision | CNN Image Classification
