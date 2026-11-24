# CIE2 – Activity 2: Binary Classification

## Project Overview
This project implements a Binary Classification model using Logistic Regression in Python to predict whether a patient has diabetes based on diagnostic health indicators. It includes data preprocessing with leakage-free imputation, stratified splitting, model training, confusion matrix analysis, and threshold evaluation.

## Objectives
- Understand Binary Classification and Logistic Regression.
- Handle missing values and scale features without data leakage.
- Understand the sigmoid activation function.
- Train a Logistic Regression model and interpret learned coefficients.
- Evaluate the model using confusion matrix, accuracy, precision, recall, specificity, F1-score, and ROC-AUC.
- Compare model performance with a majority-class baseline.
- Analyze the trade-off across different decision thresholds (0.3, 0.5, 0.7) and recommend a suitable threshold.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy / Scikit-learn
- Google Colab / Jupyter Notebook

## Dataset
- **Dataset:** Pima Indians Diabetes Dataset
- **Dataset Source:** https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv
- **Classes:** 0 (Non-Diabetic, 500 records), 1 (Diabetic, 268 records)
- **Features:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

## Work Completed
- Dataset loading, inspection, and class balance analysis.
- Identified and handled missing values encoded as zeros.
- Stratified 80:20 training and testing split.
- Imputed physiological zeros using training medians and standardized features.
- Trained an L2-regularized Logistic Regression model.
- Interpreted learned coefficients and odds ratios.
- Evaluated test performance using confusion matrix, precision, recall, F1-score, and ROC-AUC.
- Compared results with the majority-class baseline.
- Evaluated performance at thresholds 0.3, 0.5, and 0.7.
- Visualized decision behaviour and predicted outcome for a new patient profile.

## Model Evaluation Results

| Metric | Result (Threshold = 0.5) | Result (Recommended Threshold = 0.3) |
|---|---:|---:|
| Accuracy | 0.7532 | 0.7273 |
| Precision | 0.6667 | 0.5750 |
| Recall | 0.5926 | 0.8519 |
| Specificity | 0.8400 | 0.6600 |
| F1-Score | 0.6275 | 0.6866 |
| ROC-AUC | 0.8350 | 0.8350 |

## Decision Log
In diabetes screening, missing a diabetic patient (False Negative) is much more dangerous than referring a healthy patient for a follow-up test (False Positive). Therefore, I chose a decision threshold of 0.3 instead of 0.5, which improved the recall from 59.26% to 85.19% and achieved the highest F1-score (0.6866).

## Submission Details
- **Submission Profile:** [https://github.com/complexiota](https://github.com/complexiota)

## Repository Contents
- `Activity_2_Binary_Classification.ipynb` - Complete executable notebook with outputs
- `pima-indians-diabetes.csv` - Dataset file
- `Classification_Evaluation_Report.md` - Classification metrics and threshold evaluation report
- `README.md` - Project documentation
