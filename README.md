# Leadership Analytics – Employee Attrition Prediction 

Live Demo: 
https://leadership-analytics-employee-attrition-kvxizwpmeezc33u3jucq4c.streamlit.app/


A Machine Learning project developed to analyze employee and leadership-related data in order to predict employee attrition risk and support data-driven HR decision-making.

---

##  Project Overview

This project focuses on predicting employee attrition using leadership analytics and machine learning techniques.

The system analyzes employee satisfaction, work-life balance, leadership indicators, and organizational factors to identify employees at risk of leaving the company.

The project includes:
- Data Cleaning
- Data Encoding
- Feature Engineering
- Multiple Machine Learning Models
- Performance Evaluation
- Visualization & Insights

---

##  Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

##  Data Preprocessing

### Data Cleaning
- Removed missing values
- Checked data consistency
- Validated all features

### Encoding
Categorical features were transformed using Label Encoding.

Encoded Features:
- Department
- JobRole

---

## Feature Engineering

A custom feature was created:

### Leadership_Gap
Difference between:
- Performance Rating
- Job Satisfaction

This feature helps identify employees who perform well but may have low satisfaction levels, potentially indicating leadership or management issues.

---

## Machine Learning Models

Three machine learning models were implemented:

### Logistic Regression
- Simple and interpretable baseline model
- Useful for understanding feature relationships

### Random Forest Classifier
- High prediction accuracy
- Provides feature importance analysis

### Support Vector Machine (SVM)
- Effective for classification tasks
- Captures complex decision boundaries

---

##  Features Used

- Age
- Department
- Job Role
- Job Satisfaction
- Monthly Income
- Training Times Last Year
- Years at Company
- Work Life Balance
- Performance Rating
- Leadership_Gap

---

##  Model Training

- Dataset split:
  - 80% Training
  - 20% Testing

- Applied:
  - Feature Scaling
  - Hyperparameter Optimization
  - GridSearchCV

---

##  Model Evaluation

Models were evaluated using:

- Accuracy Score
- Mean Squared Error (MSE)
- Classification Report
- Confusion Matrix

Visualizations included:
- Feature Importance Graphs
- Confusion Matrix Heatmaps

---

##  Key Insights

- Random Forest achieved the best prediction performance.
- Important factors influencing attrition included:
  - Job Satisfaction
  - Work Life Balance
  - Years at Company
  - Leadership_Gap

- Logistic Regression provided interpretable results.
- SVM captured more complex employee behavior patterns.

---

## Project Goals
. Predict employee attrition risk
. Improve employee retention
. Support HR analytics
. Identify leadership-related issues
. Enable data-driven organizational decisions

---

##  How to Run

### 1. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl

```

---

---

<div align="center">

## 💙 Developed by Coding Hub

HR Analytics & Machine Learning Project

© 2026 Coding Hub. All Rights Reserved.

</div>
