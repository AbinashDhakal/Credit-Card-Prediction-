# Credit Card Fraud Detection
This repository focuses on Credit Card Fraud Detection using machine learning techniques. The dataset used includes credit card transactions in September 2013, primarily from European cardholders. With 284,807 transactions over two days, it contains 492 identified fraud cases (0.172% of all transactions). Due to confidentiality, the original features are undisclosed, except for 'Time' and 'Amount'. Features V1 to V28 result from PCA transformation.

## Overview
Explores and visualises the dataset using Pandas, Seaborn, and Matplotlib.
Standardises the 'Amount' column and handles duplicate entries for data preprocessing.
Splits the dataset into training and test sets.
Implements Logistic Regression, Decision Tree Classifier, and Random Forest Classifier for model evaluation.
Addresses the challenges of imbalanced datasets through undersampling and oversampling using SMOTE.
Visualises confusion matrices for model performance assessment.
## Dataset Information
Transactions: 284,807
Fraud Cases: 492 (0.172%)
Features: Time, Amount, V1 to V28 (PCA transformed)
Class: 1 for fraud, 0 for non-fraud
## Model Evaluation
Utilises accuracy, precision, recall, and F1-score for performance metrics.
Compares Logistic Regression, Decision Tree Classifier, and Random Forest Classifier.
