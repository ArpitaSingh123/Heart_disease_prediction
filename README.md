# 🫀 Predicting Heart Disease Risk using Multiple ML Algorithms

This project aims to predict the risk of heart disease using patient health data. Multiple machine learning algorithms were implemented in **R** to analyze and compare model performance, helping identify the most effective method for early risk prediction.

---

## 📌 Project Overview

- 🔍 Used health indicators such as age, cholesterol level, blood pressure, and family history to predict heart disease risk.
- 📊 Applied and compared the performance of the following ML algorithms:
  - **Random Forest**
  - **K-Nearest Neighbors (KNN)**
  - **K-Means Clustering**
  - **Hierarchical Clustering**
- 🛠️ Developed using **R** and libraries like `caret`, `randomForest`, `class`, `cluster`, and `factoextra`.

---

## 🧠 Machine Learning Models

### Supervised:
- **Random Forest**: For classification based on ensemble decision trees.
- **KNN**: To classify observations based on proximity to labeled training data.

### Unsupervised:
- **K-Means Clustering**: To identify natural groupings in the data.
- **Hierarchical Clustering**: To create a tree-based representation of data similarity.

---

## 📊 Evaluation Metrics

The supervised models were evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-Score  

For unsupervised models:
- Silhouette Score  
- Cluster visualization using PCA

---

## 📂 Dataset

- Based on the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- Features include:
  - Age
  - Sex
  - Resting Blood Pressure
  - Serum Cholesterol
  - Fasting Blood Sugar
  - Max Heart Rate
  - Exercise-Induced Angina
  - Oldpeak (ST depression)
  - Target (0: No Disease, 1: Disease)

---

## 🚀 How to Run

1. Clone the repository  
2. Open the `.R` or `.Rmd` script in RStudio  
3. Install required libraries:
   ```R
   install.packages(c("caret", "randomForest", "class", "cluster", "factoextra", "e1071"))
