# Credit Card Fraud Detection

Project for **detecting fraudulent credit-card transactions**:
- Data prep → EDA → modeling → threshold tuning → evaluation → **local deployment** with saved models.

> Main notebook: `fraud_detection.ipynb`  
> Local app & templates: `app.py`, `templates/`  
> Saved artifacts for deployment: `rob_scal.pkl`, `xgb_deploy.pkl`  
> Requirements: `requirement.txt`

---

## Objectives
- Build an accurate **binary classifier** to flag suspicious transactions.
- Prioritize **Recall/Precision on the Fraud class** (cost-sensitive setting).
- Provide an **inference-ready pipeline** (scaler + model) for production trials.

---

## Data & Features
- **Dataset**: Credit-card transactions (highly **imbalanced**: fraud ≪ non-fraud).
- **Targets**: `fraud` (1) vs `legit` (0).
- **Preprocessing** (typical steps used here):
  - Missing/invalid handling, encoding if needed.
  - **Scaling** with a robust scaler (artifact: `rob_scal.pkl`).
  - **Class imbalance** treatment (e.g., class_weight / resampling / threshold tuning).
- **Train/Test split** with stratification.

---

## EDA Highlights
- Inspect class imbalance (fraud rate).
- Compare distributions / correlations of key features.
- Check leakage risks and feature stability (optional drift checks).

---

## Modeling
- Baseline: simple tree/logistic for calibration.
- **Primary model**: **XGBoost (gradient boosting)** with tuned hyperparameters (robust to imbalance, nonlinearities).
- **Threshold tuning** to trade-off **Recall vs Precision** given business costs.

**Saved artifacts**  
- `rob_scal.pkl` – prefit scaler for inference  
- `xgb_deploy.pkl` – trained model for deployment  

---

## Evaluation
- **Confusion Matrix** (focus on FN & FP)
- **ROC-AUC** and **PR-AUC**
- **Recall / Precision / F1** on the Fraud class
- **Cost-aware threshold** (optional expected-cost curve)

---

## Result (XGBoost: Best Recall & Precision)
Confusion Matrix
- TN = 56,852 (True Negative) The model predicted that it was not a fraud and it was not = Correct
- FP = 12 (False Positive) The model predicted that it was a fraud but it was not = Costing time
- FN = 17 (False Negative) The model predicted that it was not a fraud but it was actually a fraud = Very risky
- TP = 81 (True Positive) The model predicted that it was a fraud and it was = Our goal!

Operational view
- False Positive Rate (FPR) ≈ 12 / 56,864 ≈ 0.021%
- Fraud Recall = 82.65% → Missed 17 out of 98 fraud cases
- Fraud Precision = 87.10% → Most of the cases flagged as fraud by the model were actually frauds.

Table
|         Class |  Precision |   Recall   |  F1-score  | Support |
| ------------: | :--------: | :--------: | :--------: | ------: |
|     0 (Legit) |   0.9997   |   0.9998   |   0.9997   |  56,864 |
| **1 (Fraud)** | **0.8710** | **0.8265** | **0.8482** |  **98** |

---

## Conclusion

1. **Best Recall & Precision: XGBoost**
  - Test and tuning 3 difference model (logistic, xgboost, ANN), validate using recall & precision.

2. **Inference-Ready**
  - The model and scaler are saved as a file. (xgb_deploy.pkl, rob_scal.pkl) and can be immediately tested on new data via app.py.

3. **Suitable for imbalanced fraud problems**
  - At a threshold of 0.20, the model yields a Recall of 82.65% and a Precision of 87.10%, with a very low FPR.
  - Reduces the risk of **Failure (FN)** without increasing the **Failure (FP)** unnecessarily.

4. **Threshold = Business Lever**
  - Want to **capture more** → Lower the threshold (Recall ↑, FP ↑)
  - Want to **control the review load** → Increase the threshold (Precision ↑, Recall ↓)

5. **Path to Production**
  - Monitoring (drift/alerts), Re-training & re-calibration, **Explainability (SHAP)** for analysts, and **audit trail/privacy compliance**

---

## Executive Summary

We built an end-to-end credit-card fraud detection pipeline—from data prep and modeling (XGBoost) to threshold tuning and deployment artifacts. 
At threshold 0.20, the model achieves Recall (Fraud) 82.65%, Precision 87.10%, and an extremely low False Positive Rate (~0.021%), balancing risk of missed fraud and review workload.
The solution is ready for inference, and with cost-based threshold tuning, monitoring, and explainability, it can be moved toward real-world operations.


