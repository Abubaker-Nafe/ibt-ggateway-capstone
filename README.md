# Capstone Project — Early Warning for High School Dropout: A Machine Learning Approach Using the HSLS:09 Longitudinal Study

## 1) Overview

**Goal:** Build a machine learning model to predict whether a student is at risk of dropping out of high school using HSLS:09 student-level data, so schools and policymakers can identify at-risk students early and intervene effectively.

**Track:** Supervised  
**Primary Metric:** Recall — because missing an at-risk student (false negative) is more costly than a false alarm (false positive) in an early-intervention setting.

## 2) Team

| Name | Primary Roles | Secondary Roles |
|---|---|---|
| Nafe Abubaker | Project Coordinator, Data Lead | EDA Lead, GitHub Lead |
| Tala Hliqawi | Slides Lead, Modeling Lead | Documentation Lead |
| Ayham AlDuwairi | Modeling Lead, Documentation Lead | GitHub Lead |
| Afaf Amwas | Project Coordinator, Data Lead | EDA and Visuals Lead |
| Heba Qasim | Data Cleaning Lead | Documentation Lead |

## 3) Dataset

- **Source:** [HSLS:09](https://nces.ed.gov/datalab/onlinecodebook/session/codebook/4d7df048-94e7-4d1f-94da-9073be961786) (High School Longitudinal Study of 2009), National Center for Education Statistics (NCES), U.S. Department of Education
- **What one row represents:** One individual student in the HSLS study
- **Population:** A nationally representative cohort of 9th-grade students first surveyed in fall 2009, with follow-up surveys in 2012, 2013, and 2016
- **Size:** ~15,900 students across ~940 high schools in 50 states and D.C.
- **Target:** `X4EVERDROP` — binary indicator (0 = never dropped out, 1 = had a known dropout episode by the 2016 follow-up)
- **Class distribution:** ~84.6% not dropout (~13,457 students) vs. ~15.4% dropout (~2,443 students) — roughly a 5.5:1 imbalance

### Datasets Used

| File | Description | Features |
|---|---|---|
| `Baseline_Dataset50.csv` | 50 features from **9th-grade (base-year)** data only | 50 |
| `Wave1_FeatureSelected.csv` | Feature-selected and engineered columns from the enhanced dataset | Selected subset |

## 4) Repository Structure

```
├── notebooks/          ← Colab notebooks (EDA, modeling)
├── data/               ← Dataset instructions (no large raw data)
├── reports/            ← Figures, charts, exported results
├── slide-deck/         ← Final presentation
├── CONTRIBUTING.md
└── README.md
```

## 5) Project Plan (Week-wise)

- **Week 7:** Proposal + dataset finalization + repo setup
- **Weeks 8–11:** EDA → baseline → improvements → polishing
- **Week 12:** Finalize repo + slides + submission

## 6) Results

### Final Model Comparison

| # | Model | Threshold | Recall | F1 | ROC-AUC | Precision |
|---|---|---:|---:|---:|---:|---:|
| 1 | Baseline — Logistic Regression | 0.50 | 0.1370 | 0.2190 | 0.7835 | 0.5447 |
| 2 | Baseline — Random Forest | 0.50 | 0.0736 | 0.1328 | 0.7838 | 0.6792 |
| 3 | Baseline — XGBoost | 0.50 | 0.1922 | 0.2848 | 0.7759 | 0.5497 |
| 4 | Baseline — LightGBM | 0.50 | 0.1840 | 0.2727 | 0.7723 | 0.5263 |
| 5 | Baseline+Wt — LR | 0.50 | 0.7280 | 0.4484 | 0.7828 | 0.3239 |
| 6 | Enhanced — Logistic Regression | 0.50 | 0.7178 | 0.4512 | 0.7839 | 0.3290 |
| 7 | Enhanced — XGBoost | 0.30 | 0.8160 | 0.3968 | 0.7687 | 0.2622 |
| 8 | Tuned XGBoost | 0.50 | 0.7137 | 0.4512 | 0.7809 | 0.3299 |
| **9** | **Tuned XGBoost** | **0.30** | **0.8650** | **0.3733** | **0.7809** | **0.2380** |
| 10 | Calibrated XGBoost | 0.30 | 0.4867 | 0.4626 | 0.7816 | 0.4407 |

> The full comparison table (17 model configurations) is available in the modeling notebook.

### Cross-Validation Results (5-Fold Stratified)

| Model | ROC-AUC | F1 | Recall |
|---|---:|---:|---:|
| Baseline — LR | 0.7907 ± 0.011 | 0.2601 ± 0.010 | 0.1663 ± 0.007 |
| Enhanced — LR | 0.7883 ± 0.009 | 0.4404 ± 0.005 | 0.7175 ± 0.010 |
| Enhanced — XGB | 0.7780 ± 0.012 | 0.4439 ± 0.019 | 0.6100 ± 0.029 |
| Tuned XGB | 0.7908 ± 0.011 | 0.4444 ± 0.016 | 0.7185 ± 0.026 |

### Recommended Model

**Tuned XGBoost at threshold = 0.30** — catches ~87% of actual dropouts (423 out of 489 in the test set). While this comes at the cost of more false alarms (precision = 0.24), the ability to identify nearly 9 out of 10 at-risk students represents the most impactful outcome for school counselors and administrators.

For deployment contexts where counselor capacity is limited, raising the threshold to 0.40–0.45 would reduce the number of flagged students while still maintaining substantially higher recall than the baseline.

## 7) Key Findings

- **Class weighting is the single biggest lever for recall.** The ablation study showed that adding class weights to the baseline Logistic Regression alone boosted recall from 0.14 to 0.73 — a +59 percentage point jump — on the same 9th-grade features. This confirms that the default threshold on imbalanced data was the primary bottleneck, not the model or features.

- **Enhanced features provide modest additional gains beyond class weighting.** Once class weights were applied, switching from the baseline to the enhanced dataset yielded smaller incremental improvements, demonstrating that the enhanced feature selection and engineering refined predictions but the weighting strategy drove the dominant recall improvement.

- **Threshold tuning unlocks the highest recall.** Lowering the decision threshold from 0.50 to 0.30 on the Tuned XGBoost pushed recall from 0.71 to 0.87. A systematic threshold sweep confirmed that the optimal F1 point sits at t≈0.61 (F1 = 0.47, Recall = 0.57), but for an early-warning system, the recall-prioritized t = 0.30 is the appropriate operating point.

- **SHAP analysis reveals the key predictors.** The most influential features for dropout prediction — identified via SHAP on the best model — provide actionable insight into which student-level factors (academic performance, engagement, socioeconomic indicators) schools should monitor most closely.

- **Calibration improves probability interpretability at the cost of recall.** Platt scaling produced well-calibrated probabilities (useful if a school wants a meaningful risk score), but shifted the probability distribution and reduced recall at the same threshold. The recommended deployment is therefore the uncalibrated Tuned XGBoost at t = 0.30 for flagging, with the calibrated version available for probability-based risk scoring.

## 8) Next Steps

- **Deploy with adjustable thresholds.** Package the model so counselors can adjust the decision threshold based on their available intervention capacity (e.g., t = 0.30 for broad screening, t = 0.45 for targeted follow-up).

- **Explore additional feature horizons.** Test whether incorporating transcript-level data or postsecondary indicators further improves prediction, and repeat the base-year vs. follow-up comparison with more granular time windows.

- **Validate on newer cohorts.** The HSLS:09 cohort entered 9th grade in 2009. Applying the model to more recent student data would test generalizability and reveal whether the same predictors remain relevant.

- **Investigate fairness and equity.** Conduct subgroup analysis across demographic categories (race/ethnicity, gender, socioeconomic status) to ensure the model does not systematically under-flag or over-flag specific populations.

- **Build a stakeholder-facing dashboard.** Create an interactive tool (e.g., Streamlit or similar) that allows school administrators to input student-level data and receive risk scores, SHAP-based explanations, and recommended intervention priorities.

## 9) How to Reproduce

1. **Obtain the data:** Download the HSLS:09 public-use file from the [NCES DataLab](https://nces.ed.gov/datalab/onlinecodebook/session/codebook/4d7df048-94e7-4d1f-94da-9073be961786).
2. **Run the Cleaning notebook** in `notebooks/` to initially clean and process the raw data.
3. **Run the EDA notebook** in `notebooks/` to explore and preprocess the data.
4. **Run the Preparing for Modeling notebook** in `notebooks/` to prepare features and structure the data for modeling.
5. **Run the modeling notebook** (`Final_Modeling.ipynb`) in `notebooks/` — it handles train/test splitting, scaling, model training, tuning, calibration, and evaluation.
6. **Environment:**
   - Python 3.8+
   - Key packages: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `shap`, `matplotlib`, `seaborn`
   - Install with: `pip install pandas numpy scikit-learn xgboost lightgbm shap matplotlib seaborn`
7. **Reproducibility:** All splits and models use `random_state=42`. The scaler is fit on training data only.
