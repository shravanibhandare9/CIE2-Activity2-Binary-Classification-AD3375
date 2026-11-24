# CIE 2 Activity 2: Binary Classification with Logistic Regression
**Department:** Artificial Intelligence and Data Science | **Class:** T.Y. B.Tech  
**Course:** Regression and Classification Algorithms | **Semester:** V (Academic Year: 2026–27)  
**Institution:** Zeal Education Society's Zeal College of Engineering and Research, Narhe, Pune  
**Record No.:** ZCOER-ACAD/R/16K | **Marks:** 5  

---

- **Faculty Submission Evaluation Profile:** [https://github.com/complexiota](https://github.com/complexiota)

---

## 1. Project Overview & Professional Framing
As a clinical data scientist supporting a community health screening programme, the objective is to build a binary classification model to predict whether a patient is at high risk for type 2 diabetes (`1 = Diabetic`) or not (`0 = Non-Diabetic`). In clinical decision-making, an error is not just a number: missing a diabetic patient (False Negative) delays critical treatment and causes irreversible microvascular damage, whereas a false alarm (False Positive) merely triggers a standard confirmatory blood test. This project establishes a rigorous, leakage-free modeling pipeline, analyzes trade-offs across decision thresholds (`0.3`, `0.5`, `0.7`), and justifies a clinically safe operating point.

---

## 2. Dataset Description and Source
- **Dataset:** Pima Indians Diabetes Dataset (768 patient records, 8 numerical features, 1 binary target)
- **Original Source Citation:** National Institute of Diabetes and Digestive and Kidney Diseases / UCI Machine Learning Repository  
  Direct CSV link: [pima-indians-diabetes.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv)
- **Target Variable (`Outcome`):**
  - `0 = Non-Diabetic`: 500 patients (65.10%)
  - `1 = Diabetic`: 268 patients (34.90%)
- **Data Quality & Physiological Zeros:**
  Domain inspection revealed that missing entries were encoded as `0` in biologically impossible columns: `Glucose` (5), `BloodPressure` (35), `SkinThickness` (227), `Insulin` (374), and `BMI` (11).

---

## 3. Decision Log
| Decision | Alternatives Considered | Professional Rationale |
|:---|:---|:---|
| **Impute physiological zeros with training median** | Drop rows or impute with mean | Dropping would discard 48.7% of all data; median is resilient to heavy right-skew in clinical markers like Insulin. |
| **Stratified 80:20 split before preprocessing** | Scale/impute before splitting | Maintains identical 34.9% positive class prevalence across sets while guaranteeing zero data leakage. |
| **Prioritize Recall over raw Accuracy** | Optimize Accuracy alone | On 65% imbalanced data, accuracy masks high false negative rates; Recall measures lives saved. |
| **Recommend Decision Threshold 0.3** | Maintain default 0.5 threshold | Lowers false negatives from 22 to 8, elevating Recall from 59.3% to **85.2%** with top F1-score (**0.6866**). |

---

## 4. Theoretical Foundation: Sigmoid Activation
Linear regression is fundamentally unsuitable for classification because it produces unbounded predictions ($-\infty, +\infty$), is vulnerable to outlier shifts, and assumes constant variance. Logistic Regression models the log-odds:

$$z = w_0 + \sum_{j=1}^p w_j x_j$$

And applies the **sigmoid activation function**:

$$\sigma(z) = \frac{1}{1 + e^{-z}} = P(y = 1 \mid \mathbf{x}) \in (0, 1)$$

---

## 5. Model Training, Coefficients and Odds Ratios
Model: L2-regularized Logistic Regression ($C=1.0$, solver = L-BFGS-B, max iterations = 1000).  
**Learned Intercept ($w_0$):** $-0.861479$

| Feature Column | Standardized Coefficient ($w_j$) | Odds Ratio per SD ($\exp(w_j)$) | Clinical Interpretation |
|:---|:---:|:---:|:---|
| **Glucose** | **+1.110765** | **3.0367** | **+203.7%** odds of diabetes per 1-SD increase (dominant predictor) |
| **BMI** | **+0.582544** | **1.7906** | **+79.1%** odds of diabetes per 1-SD increase |
| **Pregnancies** | **+0.407311** | **1.5028** | **+50.3%** odds of diabetes per 1-SD increase |
| **DiabetesPedigreeFunction** | **+0.316928** | **1.3729** | **+37.3%** odds of diabetes per 1-SD increase |
| **Age** | **+0.226240** | **1.2539** | **+25.4%** odds of diabetes per 1-SD increase |
| **SkinThickness** | **+0.069972** | **1.0725** | Modest positive risk contribution |
| **Insulin** | **-0.034721** | **0.9659** | Slight negative adjustment given high Glucose collinearity |
| **BloodPressure** | **-0.114718** | **0.8916** | Slight negative adjustment holding other factors constant |

---

## 6. Confusion Matrix & Classification Evaluation Results (Threshold = 0.5)
Evaluated on the 154 unseen test observations (100 Negative, 54 Positive):

```
                   PREDICTED LABELS
                 Negative (0)   Positive (1)
ACTUAL   Neg (0)    TN = 84        FP = 16       | 100 Actual Negatives
LABELS   Pos (1)    FN = 22        TP = 32       |  54 Actual Positives
```

### Filled Classification Evaluation Results Template:
| Metric | Mathematical Formula | What It Tells the User | Test-Set Result |
|:---|:---:|:---|:---:|
| **Accuracy** | $\frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$ | Share of correct predictions; misleading on imbalanced data. | **75.32%** ($0.7532$) |
| **Precision** | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ | Of patients predicted diabetic, fraction who truly are. | **66.67%** ($0.6667$) |
| **Recall (Sensitivity)** | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | Of truly diabetic patients, fraction detected by the model. | **59.26%** ($0.5926$) |
| **Specificity** | $\frac{\text{TN}}{\text{TN} + \text{FP}}$ | Of truly non-diabetic patients, fraction correctly cleared. | **84.00%** ($0.8400$) |
| **F1-Score** | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic balance of precision and recall. | **0.6275** |
| **ROC-AUC** | Area under ROC Curve | Ability to rank positive patients above negative patients. | **0.8350** |

- **Majority-Class Baseline Benchmark:** Always predicting Class 0 achieves **64.94% accuracy** with **0% recall** for diabetics. Logistic Regression outperforms this baseline by **+10.38%** while achieving a strong **0.8350 ROC-AUC**.

*(See [Classification_Evaluation_Report.md](Classification_Evaluation_Report.md) for full report details.)*

---

## 7. Decision Threshold Comparison & Clinical Recommendation
Comparing model performance across thresholds:

| Threshold | Predicted Positive | Accuracy | Precision | Recall | Specificity | F1-Score |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.3** | **80** | **0.7273** | **0.5750** | **0.8519** | **0.6600** | **0.6866** |
| **0.5** | 48 | 0.7532 | 0.6667 | 0.5926 | 0.8400 | 0.6275 |
| **0.7** | 30 | 0.7662 | 0.8000 | 0.4444 | 0.9400 | 0.5714 |

### Final Operational Recommendation:
We recommend **Threshold 0.3**:
- Increases Recall from 59.26% to **85.19%**, detecting **46 out of 54 diabetic patients** (slashing missed cases from 22 to 8).
- Maximizes the overall **F1-score (0.6866)**.
- Directly addresses clinical cost asymmetry: false negatives carry severe health risks, while false positives only require a routine follow-up lab test.

---

## 8. Generalization & Decision Behaviour
- **Overfitting Assessment:** Training Accuracy is 77.85% vs. Test Accuracy 75.32% (gap = 2.53%); Training ROC-AUC is 0.8474 vs. Test ROC-AUC 0.8350 (gap = 0.0124). The minimal gap validates solid generalization.
- **Decision Behaviour:** Plotting risk probability against `Glucose` shows that probability crosses the 0.3 screening threshold at $\approx 112\text{ mg/dL}$ and crosses 0.5 at $\approx 136\text{ mg/dL}$.
- **New Patient Prediction:**
  - Patient Profile: Age = 42, Pregnancies = 3, Glucose = 145, BloodPressure = 76, SkinThickness = 30, Insulin = 130, BMI = 33.5, DPF = 0.45.
  - **Predicted Diabetes Risk:** **53.47%** ($0.5347$)
  - **Screening Classification (Threshold 0.3):** **Class 1 (Diabetic – Refer for Follow-up Lab Test)**

---

## 9. Limitations & Ethical Considerations
1. **Zero-Imputation Noise:** 48.7% of Insulin entries were missing and imputed using training medians.
2. **Demographic Specificity:** The cohort comprises adult Pima Indian females; findings must be re-calibrated before deploying to broader ethnic populations.
3. **Clinical Role:** Predictions are probabilistic risk assessments designed to assist screening prioritization, never to replace clinician judgment or formal blood lab tests.




