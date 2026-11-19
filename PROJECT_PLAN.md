# CP322 Machine Learning Project Plan — Bank Telemarketing (Class Imbalance Study)

## 1. Project Goal & Instructor Requirements

- **Dataset**: UCI Bank Marketing (`bank-additional-full.csv`, Moro et al., 2014).
  - Full dataset: **41,188 rows × 20 input features + target `y`** (`sep=';'`).
  - Debug subset: `bank-additional.csv` (**4,119 rows**, 10% sample) — use for fast debugging before running full experiments.
- **Core Research Question (from Instructor)**:
  > *"Only about 12% of customers say yes. Compare three ways of handling this: (1) do nothing, (2) give the rare class more weight when training, and (3) resample the training data. Report which works best and what each one costs."*
- **Experimental Matrix**: **4 Models × 3 Imbalance Strategies = 12 core model runs** (all evaluated on the exact same untouched test set).

---

## 2. Key Dataset Findings (Verified on `bank-additional-full.csv`)

| Property | Verified Value | Modeling Implication |
|---|---|---|
| **Class distribution (`y`)** | `no`: 36,548 (**88.73%**), `yes`: 4,640 (**11.27%**) — ratio **7.88 : 1** | Accuracy is misleading (predicting all `no` gives 88.73%). Focus on **PR-AUC, ROC-AUC, F1, Precision, Recall**. |
| **Exact duplicates** | **12** duplicate rows (`41,188 → 41,176` unique rows) | Drop duplicates **before** train/test split so identical rows cannot leak across train and test. |
| **Data leakage (`duration`)** | Call duration is unknown before dialing | **Must drop `duration`** for a realistic predictive model (per dataset documentation). |
| **Missing values (`"unknown"`)** | 0 `NaN`s, but **10,700 rows (25.98%)** have `"unknown"` in at least 1 categorical column | Do **not** drop rows. Keep `"unknown"` as its own category—it carries real signal (e.g., `default="unknown"` has 5.15% `yes` rate vs. 12.88% for `"no"`; `housing` & `loan` share the exact same 990 `"unknown"` rows). Note: `default="yes"` has only **3 rows** total. |
| **Sentinel value (`pdays = 999`)** | **39,673 rows (96.32%)** have `pdays = 999` (never contacted; 9.26% `yes`). Remaining 1,515 rows have `pdays ∈ [0, 27]` (63.83% `yes`). | Raw `999` distorts Logistic Regression & Neural Nets. Add binary feature `was_previously_contacted = (pdays != 999)` and transform `pdays` (e.g., replace `999` with `0` or bin it). |
| **Feature count after OHE** | 19 raw features → **62 columns** (`drop_first=False`) / **52 columns** (`drop_first=True`) | With ~41k rows ($n/p > 660$), **aggressive feature selection is not required**. |
| **Multicollinearity** | Strong correlation among macro indicators: `emp.var.rate`–`euribor3m` ($r=0.97$), `euribor3m`–`nr.employed` ($r=0.95$), `emp.var.rate`–`nr.employed` ($r=0.91$) | L2-regularized LR, NN, and tree models handle this well for prediction; note it when interpreting LR coefficients (optional VIF check in report). |

---

## 3. Shared Experimental Protocol (Lock on Day 1 to Prevent Leakage!)

> **Critical Rule**: Preprocessing (`StandardScaler`, `OneHotEncoder`) and **Resampling** must ONLY be fit on training folds—**never** on the test set or before splitting.

1. **Data Split**:
   - Deduplicate `bank-additional-full.csv` (41,176 rows).
   - **80/20 Stratified Train/Test Split** with fixed `random_state=42`:
     - **Train**: ~32,940 rows (~29,229 `no`, ~3,711 `yes`)
     - **Test**: ~8,236 rows (~7,308 `no`, ~928 `yes`) — **never resampled, locked for final evaluation only**.
   - Hyperparameter tuning uses **5-fold Stratified Cross-Validation** inside the training set (with resampling inside an `imblearn.pipeline.Pipeline` per fold).
2. **Three Imbalance Strategies** (applied to all 4 models):
   - **Strategy A — Do Nothing (Baseline)**: Train on original 88.7% / 11.3% training set with uniform weights.
   - **Strategy B — Class Weighting**: Penalize minority errors ~7.88× more:
     - Logistic Regression & Decision Tree: `class_weight='balanced'`
     - XGBoost: `scale_pos_weight = n_neg / n_pos` (~7.88) (or Random Forest: `class_weight='balanced'`)
     - Neural Network (MLP): weighted binary cross-entropy (`pos_weight` in PyTorch / `class_weight` in Keras; note `sklearn.MLPClassifier` does not natively support `class_weight`, so use PyTorch/Keras or `sample_weight` if supported—verify early!).
   - **Strategy C — Resample Training Data**: Apply **SMOTE** (or **SMOTENC** for mixed numeric/categorical data) on `X_train, y_train` only (`~32,940 → ~58,458` rows, +77.5% size). *(Optional bonus comparison: Random Undersampling `~32,940 → ~7,422` rows to contrast oversampling vs. undersampling costs.)*
3. **Standard Output Contract (every model saves these for Tanish)**:
   - `y_test_prob` (predicted probability for `y = yes`) & `y_test_pred` (at default threshold 0.5)
   - `train_time_sec` (wall-clock training time), `n_train_samples`, and best CV hyperparameters
   - Saved in a shared `results/` folder (e.g., `results/<model>_<strategy>.npz` or `.csv`)

---

## 4. Team Roles & Timeline

### Day 1 Sync (Leonie + Tanish, 1–2 hrs)
Agree on exact train/test split indices (`random_state=42`), shared preprocessing pipeline, and output file format so everyone can work in parallel immediately afterward.

### Leonie — Data Cleaning, Feature Engineering & Shared Pipeline (Days 1–2)
- Load `bank-additional-full.csv` (`sep=';'`), drop the 12 exact duplicates, drop `duration`, encode `y` (`yes=1, no=0`).
- Engineer `was_previously_contacted = (pdays != 999)` and clean `pdays`.
- Keep `"unknown"` as a valid category; build reusable `ColumnTransformer` (`StandardScaler` for numeric + `OneHotEncoder(handle_unknown='ignore')` for categorical).
- Build reusable `imblearn` resampling helper (SMOTE / SMOTENC) that only touches training folds.
- Sanity-check the pipeline with a quick **Logistic Regression** (not Linear Regression!) and **Decision Tree** on the 10% debug file to confirm zero NaNs/encoding bugs.
- Optional EDA note: correlation heatmap of the 5 macro features + brief note explaining why aggressive feature selection isn't needed (52–62 OHE cols vs. 41k rows).

### Ayaan — Model 1 (Logistic Regression) & Model 2 (Neural Network) (Days 2–4)
- **Model 1: Logistic Regression** (aligned with Moro et al., 2014):
  - Tune regularization `C` (L2) via 5-fold Stratified CV across Strategies A, B, C.
- **Model 2: Neural Network (MLP)** (aligned with Moro et al., 2014):
  - 1–2 hidden layers with ReLU + Dropout + BatchNorm/early stopping; tuned on scaled features.
  - Ensure Strategy B uses true class-weighted BCE loss (`pos_weight` in PyTorch or `class_weight` in Keras).
- Export standardized predictions, probabilities, and training times for all 6 runs (2 models × 3 strategies).

### Kaiden — Model 3 (Decision Tree) & Model 4 (XGBoost / Random Forest) (Days 2–4)
- **Model 3: Decision Tree** (aligned with Moro et al., 2014):
  - Tune `max_depth`, `min_samples_leaf` via 5-fold Stratified CV across Strategies A, B, C (prevents severe overfitting).
- **Model 4: XGBoost** *(recommended over Random Forest for clear progression from single tree → gradient boosting)*:
  - Rationale for replacing paper's SVM: RBF-SVM has $O(n^2\text{--}n^3)$ scaling on 41k rows, whereas XGBoost is faster, handles mixed tabular data strongly, and supports `scale_pos_weight` natively.
  - Tune `max_depth`, `learning_rate`, `n_estimators`, `subsample`/`colsample_bytree` across Strategies A, B, C.
- Export standardized predictions, probabilities, and training times for all 6 runs (2 models × 3 strategies).

### Tanish — Experimental Setup, Unified Evaluation & Cost Analysis (Days 1–5)
- Lock evaluation script early (can test with Leonie's Day-2 sanity-check outputs while Ayaan & Kaiden train).
- Compute across all **12 runs** on the locked test set:
  - **Ranking & Classification Metrics**: **PR-AUC (Average Precision)**, **ROC-AUC**, **F1**, **Precision**, **Recall**, Confusion Matrix.
  - **Cost Metrics (directly answering *"what each one costs"*)**:
    1. **False Positive / Marketing Call Cost**: Precision drop & extra wasted calls (FP count) required to gain extra Recall (TP count).
    2. **Compute & Data Cost**: Training time (`train_time_sec`) and training set size (`32.9k` vs. `58.5k` for SMOTE).
    3. **Probability Calibration Cost**: **Brier Score** + Calibration Curve (class weighting and resampling inflate predicted probabilities away from the true 11.3% base rate).
    4. *(Optional high-impact bonus)*: Compare against **Do Nothing + Threshold Tuning** on validation F1 to show whether weighting/resampling actually beats simply adjusting the decision threshold.
- Generate final comparison tables and plots for the report and slides.

---

## 5. Report & 5-Minute Presentation Structure

- **Report**: Each member writes their own technical section, then Tanish + team unify the final Conclusion/Recommendation section so findings are consistent.
- **5-Minute Presentation Pacing**:
  1. **0:00–0:45** — Problem & severe class imbalance (11.27% `yes`, why Accuracy fails).
  2. **0:45–1:45** — Preprocessing highlights (dropping `duration`, handling `pdays=999` & `"unknown"`, strict leakage-free split/pipeline).
  3. **1:45–3:30** — Core 4×3 benchmark results (PR-AUC, ROC-AUC, F1, Precision/Recall comparison).
  4. **3:30–5:00** — **"What each method costs"** (FP/call volume trade-off, compute time, probability calibration) + final recommendation for the bank.

