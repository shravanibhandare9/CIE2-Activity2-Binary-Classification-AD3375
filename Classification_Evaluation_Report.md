# Activity 2: Classification Evaluation Report

## 1. Problem Overview & Clinical Context
In healthcare screening for type 2 diabetes mellitus, our goal is early detection. The consequence of classification errors is asymmetric:
- **False Negative (FN):** A diabetic patient is erroneously diagnosed as healthy, missing early lifestyle intervention and pharmacotherapy. This leads to preventable long-term complications including diabetic retinopathy, nephropathy, peripheral neuropathy, and cardiovascular disease.
- **False Positive (FP):** A non-diabetic patient is flagged as positive and referred for a confirmatory laboratory test (such as an Oral Glucose Tolerance Test or glycated hemoglobin $\text{HbA}_{1c}$). The cost is minor clinical inconvenience and small testing fees.

Therefore, this evaluation focuses heavily on **Sensitivity (Recall)** and the **ROC-AUC** metric, demonstrating why the default 0.5 decision threshold must be adjusted.

---

## 2. Test-Set Confusion Matrix Analysis (Threshold = 0.5)
Evaluated on the $n = 154$ stratified holdout test set (100 non-diabetic patients, 54 diabetic patients):

```
                   PREDICTED LABELS
                 Negative (0)   Positive (1)
ACTUAL   Neg (0)    TN = 84        FP = 16       | 100 Total Actual Negatives
LABELS   Pos (1)    FN = 22        TP = 32       |  54 Total Actual Positives
                    ------         ------
                   106 Total      48 Total
```

### Breakdown of Outcomes:
- **True Positives (TP = 32):** 32 diabetic patients correctly identified for clinical intervention.
- **True Negatives (TN = 84):** 84 healthy patients correctly cleared without unnecessary anxiety.
- **False Positives (FP = 16):** 16 healthy patients referred for confirmatory laboratory testing (false alarms).
- **False Negatives (FN = 22):** 22 diabetic patients missed at threshold 0.5 (**40.7% of all diabetic cases missed**).

---

## 3. Classification Evaluation Results Template
The table below complies exactly with the CIE 2 Student Activity Instructions template:

| Metric | Mathematical Formula | What It Tells the User | Your Test-Set Result |
|:---|:---:|:---|:---:|
| **Accuracy** | $\frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$ | Share of all predictions that are correct; misleading when classes are imbalanced. | **75.32%** ($0.7532$) |
| **Precision** | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ | Of the records predicted positive, how many really are positive. | **66.67%** ($0.6667$) |
| **Recall (Sensitivity)** | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | Of the truly positive records, how many the model found. | **59.26%** ($0.5926$) |
| **Specificity** | $\frac{\text{TN}}{\text{TN} + \text{FP}}$ | Of the truly negative records, how many the model correctly cleared. | **84.00%** ($0.8400$) |
| **F1-Score** | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ | Single score balancing precision and recall via their harmonic mean. | **0.6275** |
| **ROC-AUC** | $\int_0^1 \text{TPR}(t) \, d(\text{FPR}(t))$ | Ability to rank positives above negatives across all thresholds ($0.5 = \text{chance}, 1.0 = \text{perfect}$). | **0.8350** |

*Interpretation:* The model demonstrates strong discriminative ranking power ($\text{ROC-AUC} = 0.8350$), but at the standard 0.5 decision threshold, it captures fewer than 60% of true diabetic cases ($\text{Recall} = 59.26\%$), proving that threshold calibration is mandatory.

---

## 4. Full Classification Report (Threshold = 0.5)

| Class | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **0 (Non-Diabetic)** | 0.7925 | 0.8400 | 0.8155 | 100 |
| **1 (Diabetic)** | 0.6667 | 0.5926 | 0.6275 | 54 |
| **Macro Average** | 0.7296 | 0.7163 | 0.7215 | 154 |
| **Weighted Average** | 0.7483 | 0.7532 | 0.7496 | 154 |

---

## 5. Majority-Class Baseline Comparison
- **Baseline Rule:** Predict the majority class (`0 – Non-Diabetic`) for 100% of patient records.
- **Baseline Accuracy:** $\frac{100}{154} = \mathbf{64.94\%}$
- **Baseline Recall for Positives:** $\mathbf{0.00\%}$ ($\text{TP} = 0, \text{FN} = 54$)
- **Baseline F1-Score:** $\mathbf{0.00}$

**Why Accuracy Misleads:** A naive model that always predicts healthy achieves nearly 65% accuracy while completely failing its core mission—it detects zero cases of diabetes. Logistic Regression delivers **75.32% accuracy** (+10.38 percentage points) and an **ROC-AUC of 0.8350**, establishing genuine clinical utility.

---

## 6. Threshold Comparison Template & Trade-Off Analysis
The table below complies with the CIE 2 Threshold Comparison Template:

| Threshold | Predicted Positive | Accuracy | Precision | Recall | Specificity | F1-Score |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.3** | **80** | **0.7273** | **0.5750** | **0.8519** | **0.6600** | **0.6866** |
| **0.5** | 48 | 0.7532 | 0.6667 | 0.5926 | 0.8400 | 0.6275 |
| **0.7** | 30 | 0.7662 | 0.8000 | 0.4444 | 0.9400 | 0.5714 |

### Mathematical Precision–Recall Trade-off:
- **Lowering Threshold (0.5 $\rightarrow$ 0.3):**
  - Predicted positives expand from 48 to 80 patients.
  - **Recall leaps from 59.26% to 85.19%** (detecting 46 out of 54 diabetics, cutting missed cases from 22 to 8).
  - Precision decreases from 66.67% to 57.50% (accepting 34 false alarms instead of 16).
  - **F1-Score increases to its maximum value (0.6866).**
- **Raising Threshold (0.5 $\rightarrow$ 0.7):**
  - Precision rises to 80.00%, but Recall drops drastically to 44.44% (misses 30 out of 54 diabetic patients).
  - In a screening environment, missing 55.6% of diseased patients is unacceptable.

---

## 7. Justification of Recommended Threshold ($\theta = 0.3$)
We formally recommend **Threshold 0.3** for deployment in the public health screening workflow:
1. **Asymmetric Error Costs:** False Negatives cause irreversible organ and nerve damage, whereas False Positives only result in an inexpensive confirmatory blood test.
2. **Clinical Safety:** Catches **85.2%** of all diabetic patients.
3. **Optimal Balance:** Achieves the highest test $F_1$-score ($0.6866$) among all evaluated thresholds.

---

## 8. Overfitting and Generalization Assessment
To ensure the model has not memorized training noise, metrics are compared across splits at the default threshold (0.5):

| Split | Sample Size | Accuracy | Precision | Recall | Specificity | F1-Score | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Training Set** | 614 | 77.85% | 72.94% | 57.94% | 88.50% | 0.6458 | 0.8474 |
| **Holdout Test Set** | 154 | 75.32% | 66.67% | 59.26% | 84.00% | 0.6275 | 0.8350 |
| **Generalization Gap** | — | **-2.53%** | **-6.27%** | **+1.32%** | **-4.50%** | **-0.0183** | **-0.0124** |

**Conclusion:** The generalization gap in accuracy is only 2.53%, and the ROC-AUC gap is merely 0.0124. This confirms excellent generalization without overfitting or underfitting, validated by the L2 regularization penalty ($C=1.0$).


