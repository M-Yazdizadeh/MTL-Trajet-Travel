# **MTL-Trajet Travel Mode Analysis**

This repository contains data analysis, preprocessing, and machine learning workflows for classifying and analyzing travel modes using the **MTL-Trajet** dataset. The project focuses on applying supervised learning algorithms, neural networks, and clustering techniques to predict travel modes, extract meaningful insights, and explore natural groupings.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Key Features](#key-features)
4. [Modeling Workflow](#modeling-workflow)
5. [Clustering Analysis](#clustering-analysis)
6. [Technologies Used](#technologies-used)
7. [Results and Insights](#results-and-insights)
8. [License](#license)

---

## **Project Overview**
This project leverages machine learning, deep learning, and clustering techniques to classify travel modes and uncover patterns using the **MTL-Trajet** dataset. The focus is on comparing model performances, improving accuracy, and exploring natural data clusters while addressing challenges such as class imbalance and missing data.

---

## **Dataset**
- **Name**: MTL-Trajet Travel Dataset
- **Source**: [Open Canada Dataset](https://open.canada.ca/data/en/dataset/955de068-9556-4aa1-980e-487b057bbc9d)
- **Preprocessing**:
  - Removed unnecessary columns (e.g., `id_trip`, `geometry`).
  - Handled missing values using imputation techniques.
  - Encoded travel mode labels for consistency.
  - Scaled numerical features for clustering and neural network models.

---

## **Key Features**
- Multimodal travel prediction using:
  - Neural networks (PyTorch)
  - Tree-based models (Decision Trees, Random Forests, XGBoost)
- Clustering analysis using:
  - **K-Means** for defining behavioral clusters.
  - **DBSCAN** for identifying density-based patterns and outliers.
- Addressing class imbalance with **SMOTE (Synthetic Minority Over-sampling Technique)**.
- Hyperparameter optimization using **RandomizedSearchCV**.
- Performance evaluation with detailed classification reports and visualizations.

---

## **Modeling Workflow**
1. **Preprocessing**:
   - Label encoding for categorical variables.
   - Handling missing values using imputation.
   - Feature scaling for neural networks and clustering analysis.
2. **Training Models**:
   - PyTorch Neural Network
   - Decision Tree Classifier
   - Random Forest Classifier
   - Gradient Boosting (XGBoost)
3. **Clustering Analysis**:
   - K-Means clustering on PCA-reduced data.
   - DBSCAN clustering for density-based group identification.
4. **Evaluation**:
   - Classification reports for supervised models.
   - Clustering visualization and detailed analysis of clusters.

---

## **Clustering Analysis**
### **PCA Overview**
Principal Component Analysis (PCA) was applied to reduce the dataset dimensions for clustering. Two principal components (PCA1 and PCA2) were extracted, revealing the following patterns:
- **PCA1**: Differentiates data based on travel speed and motorized travel modes (e.g., cars, motorcycles).
- **PCA2**: Highlights public transport usage and work-related trip purposes.

#### **Top Features in PCA Loadings**
- **PCA1**:
  - `mode_Voiture / Moto` (Weight: 0.5445)
  - `speed` (Weight: 0.4852)
  - `purpose_Santé` (Weight: 0.1341)
- **PCA2**:
  - `purpose_Travail / Rendez-vous d'affaires` (Weight: 0.6970)
  - `mode_Transport collectif` (Weight: 0.1496)

### **K-Means Clustering Results**
- **Clusters**:
  - **Cluster 0**: High-speed trips using motorized vehicles, predominantly for leisure and health purposes.
  - **Cluster 1**: Medium-speed trips involving walking or cycling, with a mix of purposes.
  - **Cluster 2**: Work-related trips using public transport or multi-modal travel.
- **Insights**:
  - Clear separation of clusters based on speed, purpose, and travel mode.
  - Effective for understanding behavioral patterns in multimodal travel.

### **DBSCAN Clustering Results**
- **Clusters**:
  - **Cluster 0**: Majority of trips with routine travel patterns.
  - **Cluster 1**: Smaller group of slower or geographically distinct trips.
  - **Outliers (-1)**: Unique trips or anomalies with low density.
- **Insights**:
  - DBSCAN identified outliers and natural groupings in the data.
  - Effective for capturing irregular travel patterns and unique behaviors.

---

## **Technologies Used**
- **Languages**: Python
- **Libraries**:
  - Data Processing: `pandas`, `numpy`, `scikit-learn`
  - Machine Learning: `RandomForestClassifier`, `XGBClassifier`, `DecisionTreeClassifier`
  - Deep Learning: `torch`, `torch.nn`, `torch.optim`
  - Clustering: `KMeans`, `DBSCAN`
  - Visualization: `matplotlib`, `seaborn`

---

## **Results and Insights**
### **Supervised Learning Results**
- **Best Model**: Gradient Boosting (XGBoost) achieved the highest accuracy of **97%**.
- **Neural Network**: Moderately accurate (81%) but requires further optimization.
- **Random Forest**: Robust performance (96%), generalizable for various travel patterns.

### **Clustering Results**
- **K-Means**:
  - Effectively separated clusters based on travel speed, mode, and purpose.
  - Provided actionable insights into user behavior for urban planning.
- **DBSCAN**:
  - Highlighted outliers and rare travel behaviors.
  - Useful for identifying unique commuting patterns or data anomalies.

---

## **License**
This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this code with proper attribution.

---

## **Contact**
For questions or collaborations, please reach out to **Mohammad Yazdizadeh**:
- **GitHub**: [M-Yazdizadeh](https://github.com/M-Yazdizadeh)
