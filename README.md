# Fraud Detection Model Comparison & Recommendation

## Project Overview

This project evaluates multiple machine learning models for fraud detection at Deacon Financial Services following a surge in fraudulent applications during their "New Money Bonus" campaign. The analysis prioritizes **recall** (catching fraud) while maintaining operational feasibility through controlled false positive rates.

**Final Recommendation:** Deploy **XGBoost** as the primary fraud detection engine, achieving ~80% recall with balanced precision and operational efficiency.

---

## Business Context

### Problem Statement
Deacon Financial Services experienced rapid applicant growth that triggered increased fraudulent activity. The challenge: detect maximum fraud cases (high recall) while controlling false positives to avoid unnecessary investigations and protect customer experience.

### Key Priorities
- **Maximize fraud detection** (high recall)
- **Control operational burden** (manageable false positives)
- **Balance accuracy with interpretability** for governance and compliance

---

## Model Performance Summary

| Model | Accuracy | ROC-AUC | Precision | Recall | F1 Score | Recommendation |
|-------|----------|---------|-----------|--------|----------|----------------|
| **XGBoost** ⭐ | 0.676 | 0.821 | 0.0026 | **0.798** | 0.0052 | **DEPLOY** |
| Logistic Regression | 0.820 | 0.823 | 0.0038 | 0.654 | 0.0076 | Support/Governance |
| Ridge Regression | 0.991 | 0.831 | 0.0197 | 0.160 | 0.035 | ❌ Misses 84% of fraud |
| Elastic Net | 0.001 | 0.836 | 0.0011 | 1.000 | 0.0021 | ❌ Excessive false positives |
| Decision Tree | 0.620 | 0.751 | 0.0022 | 0.805 | 0.0045 | ❌ Unstable |
| Random Forest | 0.671 | 0.738 | 0.0022 | 0.673 | 0.0043 | ❌ Lower ROC-AUC |

---

## Key Findings

### 1. **Recall vs. Precision Trade-off**
- **Elastic Net:** Perfect recall (100%) but unusable due to extreme false positives
- **Ridge:** High precision but catastrophically low recall (16%)
- **XGBoost:** Optimal balance - captures 80% of fraud with manageable alerts

### 2. **Accuracy is Misleading**
Due to severe class imbalance (~0.1% fraud rate), high accuracy doesn't equal business value. Ridge achieves 99% accuracy but misses most fraud cases.

### 3. **Interpretability vs. Performance**
Linear models provide transparency but fail to capture complex fraud patterns. XGBoost's superior detection capability outweighs interpretability concerns, which can be addressed through SHAP values and secondary explainable models.

---

## Deployment Strategy

### Primary Architecture
```
┌─────────────────────────────────────┐
│   XGBoost Primary Detection Layer   │
│   • Recall: ~80%                    │
│   • Threshold: 0.006506             │
│   • ROC-AUC: 0.821                  │
└─────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Logistic Regression Support Layer  │
│   • Explainability & Governance     │
│   • Regulatory Reporting            │
│   • Threshold Validation            │
└─────────────────────────────────────┘
```

### Implementation Phases

**Phase 1: Controlled Rollout**
- Deploy XGBoost in shadow mode alongside existing systems
- Monitor recall, precision, and false positive rates
- Validate threshold calibration (current: 0.006506)

**Phase 2: Production Deployment**
- Activate XGBoost as primary detection engine
- Implement real-time scoring pipeline
- Establish alert routing to investigation teams

**Phase 3: Continuous Improvement**
- Weekly performance monitoring
- Monthly model retraining with new fraud patterns
- Quarterly threshold recalibration based on cost analysis

---

## Repository Structure

```
fraud-detection/
│
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned and feature-engineered data
│   └── results/                # Model predictions and evaluations
│
├── models/
│   ├── xgboost_final.pkl       # Production XGBoost model
│   ├── logistic_support.pkl    # Interpretable support model
│   └── config/
│       └── hyperparameters.yaml
│
├── notebooks/
│   ├── 01_eda.ipynb                      # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb      # Feature creation
│   ├── 03_model_comparison.ipynb         # All model evaluations
│   └── 04_threshold_optimization.ipynb   # Business-cost calibration
│
├── src/
│   ├── preprocessing.py        # Data cleaning pipeline
│   ├── train.py                # Model training scripts
│   ├── evaluate.py             # Performance metrics
│   └── deploy.py               # Inference pipeline
│
├── reports/
│   ├── final_recommendation.pdf         # Full analysis report
│   ├── model_performance_appendices.pdf # Detailed metrics
│   └── business_impact_analysis.pdf     # ROI projections
│
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── LICENSE
```

---

## Technical Requirements

### Dependencies
```
python>=3.8
xgboost>=1.7.0
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
shap>=0.41.0  # For model explainability
```


## Usage

### Training Models
check attached Model under Machine Learning



## Model Performance Details

### XGBoost (Recommended)
```
Optimal Threshold: 0.006506
Accuracy:  67.58%
ROC-AUC:   82.08%
Precision: 0.26%
Recall:    79.77%
F1 Score:  0.52%

Confusion Matrix:
[[164,137  78,797]  ← 32% false positive rate on non-fraud
 [    52     205]]  ← 80% fraud detection rate

Business Impact:
• Detects 205 of 257 fraud cases (79.77%)
• Generates 78,797 alerts for investigation
• Alert-to-fraud ratio: ~384:1
```

### Why Not Other Models?

**Ridge Regression** (Highest Accuracy: 99.07%)
- Misses 84% of fraud cases (41 of 257 detected)
- Unacceptable business risk despite high accuracy

**Elastic Net** (Perfect Recall: 100%)
- Flags every application as fraud
- 242,934 false positives = operationally impossible

**Logistic Regression** (Moderate Performance)
- Only 65% recall - misses 35% of fraud
- Better than Ridge but inferior to XGBoost

---

## Business Impact

### Projected Outcomes
- **Fraud Loss Reduction:** ~80% of fraudulent applications caught
- **Investigation Efficiency:** Manageable alert volume through threshold tuning
- **Customer Experience:** Minimized false positives compared to Elastic Net
- **Compliance:** Explainable secondary layer for regulatory requirements

### Cost-Benefit Analysis
Assuming:
- Average fraud loss: $1,000 per case
- Investigation cost: $50 per alert

**XGBoost Performance:**
- Fraud prevented: 205 cases × $1,000 = **$205,000 saved**
- Investigation cost: 78,797 alerts × $50 = **$3,939,850**
- Net impact depends on threshold optimization

**Threshold Tuning Opportunity:** Adjust threshold to optimize cost-benefit ratio based on actual investigation capacity and fraud costs.

---

## Monitoring & Maintenance

### KPIs to Track
- **Recall:** Target ≥75% (never drop below 70%)
- **Precision:** Monitor trend, aim to improve without sacrificing recall
- **False Positive Rate:** Target <35% of legitimate applications
- **ROC-AUC:** Maintain >0.80

### Retraining Schedule
- **Weekly:** Performance monitoring
- **Monthly:** Model retraining with new data
- **Quarterly:** Full model reevaluation and threshold recalibration

### Alerts & Escalation
- Recall drops below 70% → Immediate investigation
- Precision deteriorates >20% → Review threshold and features
- New fraud patterns detected → Accelerate retraining cycle

---

## Contributors

**Analysis Lead:** Jim Appiah  
**Stakeholder:** Deacon Financial Service Management  (Hypothetical)
**Date:** January 23, 2026

---



## Next Steps

1. **Immediate:** Deploy XGBoost in controlled production environment
2. **Week 1:** Establish monitoring dashboards and alert pipelines
3. **Month 1:** Analyze production performance and tune threshold
4. **Quarter 1:** Implement SHAP-based explainability layer for compliance
5. **Ongoing:** Continuous model improvement and fraud pattern analysis

---

## Contact

For questions or access requests:  
appiahjim024@gmail.com
