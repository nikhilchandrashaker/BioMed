# BioMed — Biomedical Machine Learning Pipelines

A collection of self-contained Python pipelines that apply classic and
ensemble ML methods to classic UCI / clinical biomedical datasets. Every
pipeline follows the same shape: **load → clean/encode → cross-validate →
visualize**, and each one saves a single, dense summary figure (ROC curves,
confusion matrix, feature importances, per-fold scores, class balance, and a
metrics table) to make results easy to scan and compare across diseases.

## Why this structure

Small clinical datasets (dozens to a few hundred patients) are exactly the
setting where the *choice of validation strategy matters as much as the
model*. These pipelines deliberately lean on:

- **Stratified K-Fold CV** — the default workhorse, keeps class balance per fold.
- **Leave-One-Out CV** — near-unbiased estimate for very small `n`, at the cost of compute.
- **Group-aware CV** (Parkinson's) — prevents leakage when a dataset has repeated measurements per patient.
- **SMOTE** (gallstone, cervical cancer) — for pipelines with meaningful class imbalance.
- **Gradient Boosting / Random Forest / SVM ensembles** — strong, interpretable-enough defaults for tabular clinical data, always inspected via feature importances.

## Pipelines in this collection

Every pipeline below has been run end-to-end on real data; the AUC figures are
from those runs (see each PNG for the full breakdown).

| Pipeline | Dataset | n | Features | Target | Techniques | ROC-AUC (SKF / LOO) |
|---|---|---|---|---|---|---|
| `breast_cancer_pipeline.py` | UCI Breast Cancer (Ljubljana) | 286 | 9 | recurrence-events vs no-recurrence | Stratified K-Fold, LOO, Gradient Boosting | 0.647 / 0.649 |
| `hepatitis_pipeline.py` **(new)** | UCI Hepatitis Domain | 155 | 19 | DIE vs LIVE | Stratified K-Fold, LOO, Gradient Boosting, median imputation | 0.828 / 0.836 |
| `parkinsons_pipeline.py` **(new)** | UCI Oxford Parkinson's Disease Detection | 195 recordings / 32 subjects | 22 voice measures | healthy vs Parkinson's | Stratified K-Fold, **Subject-Grouped K-Fold** (leak check), LOO, Gradient Boosting | 0.967 naive / **0.710 subject-grouped** / 0.974 LOO |
| `heart_failure_pipeline.py` | Heart Failure Clinical Records | 299 | 12 | DEATH_EVENT | Stratified K-Fold, LOO, Gradient Boosting | 0.888 / 0.894 |
| `chronic_kidney_disease_pipeline.py` | UCI Chronic Kidney Disease | 400 | 24 | ckd vs notckd | Stratified K-Fold, LOO, Random Forest | 1.000 / 1.000 |
| `lung_cancer_pipeline.py` | UCI Lung Cancer | 32 | 56 | cancer type (3-class) | LOO (small-n), Stratified K-Fold, SVM + RF ensemble, imputation | 0.699 macro (LOO) |
| `diabetes_pipeline.py` | UCI Diabetes Log Data | 70 patients | engineered from time-series | hypoglycemic episode | Time-series event feature engineering, Stratified K-Fold, LOO, Gradient Boosting | 0.817 / 0.753 |
| `gallstone_pipeline.py` | UCI Gallstone | 319 | 38 | gallstone status | Stratified K-Fold, SMOTE, Random Forest | 0.862 (SKF OOF) |
| `cervical_cancer_pipeline.py` | UCI Risk Factors Cervical Cancer | 858 | 36 | Biopsy, other 3 diagnostic tests dropped to avoid leakage | Stratified K-Fold, SMOTE, Random Forest | 0.595 (SKF OOF) |

A per-pipeline summary figure (ROC curves, confusion matrix, feature
importances, per-fold AUC, class balance, metrics table) is saved next to
each script — e.g. `chronic_kidney_disease_pipeline.png`.

### Notes on a few results

- **CKD ≈ 1.000 AUC** is not a bug — this is a well-known property of the UCI CKD dataset: `ckd` and `notckd` are almost perfectly separable on lab values like hemoglobin and specific gravity, so near-perfect scores are expected and widely reported in the literature.
- **Cervical cancer ≈ 0.60 AUC** looks unremarkable next to papers reporting >0.9, but those numbers typically include `Hinselmann`, `Schiller`, and `Citology` — three other diagnostic tests — as *input features* to predict `Biopsy`, which is a form of leakage (using test results to predict a closely related test result). This pipeline deliberately drops all three and predicts `Biopsy` from lifestyle/history risk factors alone, which is the harder and more clinically meaningful version of the task.
- **Breast cancer ≈ 0.65 AUC** on 9 coarse, binned categorical features (age brackets, tumor-size brackets, etc.) is in line with published baselines for this specific dataset — it's a genuinely hard, low-signal classic dataset, not a broken pipeline.

## New in this batch

### `hepatitis_pipeline.py`
Predicts short-term mortality (**DIE**, coded as the positive class 1) vs
survival (**LIVE**) from 19 clinical/lab variables — bilirubin, albumin,
prothrombin time, liver exam findings, ascites, varices, etc. About 8% of
cells are missing (`?` in the raw file), concentrated in a handful of lab
tests, so the pipeline uses **median imputation** on a mix of continuous labs
and 0/1-recoded categorical findings. Bilirubin, albumin, and prothrombin
time dominate feature importance — consistent with their clinical role as
liver-synthetic-function markers.

- Stratified 10-Fold ROC-AUC ≈ 0.83, Leave-One-Out ROC-AUC ≈ 0.84 on the run used to build the figure.
- The 155-patient cohort is imbalanced (123 LIVE / 32 DIE), so precision/recall/F1 are reported alongside AUC, not just accuracy.

### `parkinsons_pipeline.py`
Classifies sustained-vowel voice recordings as **healthy** or
**Parkinson's** using 22 dysphonia measures (jitter, shimmer, noise-to-harmonics
ratios, nonlinear complexity measures like RPDE/DFA/PPE). This dataset has
~6 recordings per patient (195 recordings from 32 subjects), and the
pipeline is built specifically to surface — not hide — the resulting risk of
**data leakage**:

- A naive Stratified K-Fold (recording-level, ignoring patient identity) reports ROC-AUC ≈ 0.97 — because recordings from the same patient can appear in both train and test folds.
- A **Subject-Grouped K-Fold** (`GroupKFold` keyed on the parsed patient id) drops to ROC-AUC ≈ 0.71 — the honest estimate of how the model performs on a genuinely new patient.
- Leave-One-Out (recording-level) is also reported for completeness but inherits the same leakage caveat as the naive K-Fold.

The output figure plots all three ROC curves together and calls out the gap
explicitly in the validation-summary table, so the leakage effect is visible
rather than silently inflating headline numbers.

## Running a pipeline

Each script expects its source data at a fixed local path near the top of
the file (mirroring how the original UCI files are typically staged) —
update that path if your data lives elsewhere, e.g.:

```python
# hepatitis_pipeline.py
df = pd.read_csv('/tmp/hepatitis_data/hepatitis.data', header=None, names=COLS, na_values='?')

# parkinsons_pipeline.py
df = pd.read_csv('/tmp/parkinsons_data/parkinsons.data')
```

Then just run it:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
python hepatitis_pipeline.py
python parkinsons_pipeline.py
```

Each script prints CV metrics to stdout and saves one PNG summary figure
(e.g. `hepatitis_pipeline.png`, `parkinsons_pipeline.png`) — no notebook or
GUI required (`matplotlib.use('Agg')`).

## Data sources

All datasets originate from the UCI Machine Learning Repository unless noted;
raw files were sourced from public GitHub mirrors of the original UCI
releases where a local copy wasn't already on hand.

- **Hepatitis**: G. Gong (Carnegie Mellon) via B. Cestnik, Jozef Stefan Institute, 1988. Cost-of-testing metadata (`hepatitis.cost/.delay/.expense/.group`) donated by Peter Turney, 1995.
- **Parkinson's Disease Detection**: Little, McSharry, Hunter, Ramig (2008), *"Suitability of dysphonia measurements for telemonitoring of Parkinson's disease,"* IEEE Trans. Biomedical Engineering.
  - Parkinson's Telemonitoring (UPDRS regression variant, `parkinsons_updrs.data`, 5,875 recordings / 42 patients) is included in the raw data but not yet covered by a pipeline here — a natural next addition, predicting `motor_UPDRS` / `total_UPDRS` via regression instead of classification.
- **Breast Cancer (Ljubljana)**: Zwitter, M. & Soklic, M. (1988), Institute of Oncology, Ljubljana.
- **Heart Failure Clinical Records**: Chicco, D. & Jurman, G. (2020), BMC Medical Informatics and Decision Making.
- **Chronic Kidney Disease**: Rubini, L., Soundarapandian, P., & Eswaran, P. (2015).
- **Lung Cancer**: Hong, Z.Q. & Yang, J.Y. — donor dataset, UCI ML Repository.
- **Diabetes (time-series log data)**: Kahn, M., UCI ML Repository — 70 patients' logged blood-glucose, insulin, and lifestyle events.
- **Gallstone**: Esen, I., Arslan, H., Aktürk, S., Gülşen, M., Kültekin, N., & Özdemir, O. (2024), *Medicine*, doi: 10.1097/md.0000000000037258.
- **Cervical Cancer (Risk Factors)**: Fernandes, K., Cardoso, J., & Fernandes, J. (2017).

## Caveats

These are educational/portfolio pipelines on small, decades-old public
datasets — not validated clinical decision tools. In particular:

- Several datasets (hepatitis, lung cancer, CKD) are small enough that CV
  variance across folds is substantial (see the per-fold AUC bars in each
  figure) — treat point estimates as noisy.
- The Parkinson's pipeline is the clearest built-in reminder that **grouped,
  patient-aware validation** can matter more than the choice of model.
- None of these pipelines do hyperparameter tuning beyond fixed, reasonable
  defaults; that's a natural next step before comparing techniques further.
