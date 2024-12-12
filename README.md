
# **MTL-Trajet Travel Mode Analysis**

This repository contains data analysis, preprocessing, and machine learning workflows for classifying and analyzing travel modes using the **MTL-Trajet** dataset. The project focuses on applying supervised learning algorithms and neural networks to predict travel modes and extract meaningful insights.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Key Features](#key-features)
4. [Modeling Workflow](#modeling-workflow)
5. [Technologies Used](#technologies-used)
6. [Results and Insights](#results-and-insights)
7. [License](#license)

---

## **Project Overview**
This project leverages machine learning and deep learning techniques to classify travel modes based on features derived from the **MTL-Trajet** dataset. The focus is on comparing model performances and improving accuracy while addressing challenges such as class imbalance and missing data.

---

## **Dataset**
- **Name**: MTL-Trajet Travel Dataset
- **Source**: https://open.canada.ca/data/en/dataset/955de068-9556-4aa1-980e-487b057bbc9d.
- **Preprocessing**:
  - Unnecessary columns (e.g., `id_trip`, `geometry`) were removed.
  - Missing values were handled using imputation techniques.
  - Travel mode labels were translated and encoded for consistency.

---

## **Key Features**
- Multimodal travel prediction using:
  - Neural networks (PyTorch)
  - Tree-based models (Decision Trees, Random Forests, XGBoost)
- Addressing class imbalance with **SMOTE (Synthetic Minority Over-sampling Technique)**.
- Hyperparameter optimization using **RandomizedSearchCV**.
- Performance evaluation with detailed classification reports and visualizations.

---

## **Modeling Workflow**
1. **Preprocessing**:
   - Label encoding for categorical variables.
   - Handling missing values using imputation.
   - Feature scaling for neural networks.
2. **Training Models**:
   - PyTorch Neural Network
   - Decision Tree Classifier
   - Random Forest Classifier
   - Gradient Boosting (XGBoost)
3. **Hyperparameter Tuning**:
   - Grid search and random search for optimal parameters.
4. **Evaluation**:
   - Classification reports for precision, recall, F1-score.
   - Model comparison and result visualization.

---

## **Technologies Used**
- **Languages**: Python
- **Libraries**:
  - Data Processing: `pandas`, `numpy`, `scikit-learn`
  - Machine Learning: `RandomForestClassifier`, `XGBClassifier`, `DecisionTreeClassifier`
  - Deep Learning: `torch`, `torch.nn`, `torch.optim`
  - Oversampling: `imblearn`
  - Visualization: `matplotlib`

---

## **Results and Insights**
- **Best Model**: Gradient Boosting (XGBoost) achieved the highest accuracy of **97%**.
- **Findings**:
  - Neural networks performed moderately (81%), requiring further optimization.
  - Random Forest (96%) provided a robust and generalizable alternative.
  - Decision Tree (93%) remains interpretable but less effective than ensemble methods.
  - Proper handling of class imbalance (SMOTE) and hyperparameter tuning significantly improved results.

---

## **License**
This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this code with proper attribution.

---

## **Contact**
For questions or collaborations, please reach out to **Mohammad Yazdizadeh**:
- **GitHub**: [M-Yazdizadeh](https://github.com/M-Yazdizadeh)
