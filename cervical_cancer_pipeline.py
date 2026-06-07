"""
=============================================================
  CERVICAL CANCER RISK CLASSIFICATION PIPELINE
  Dataset: UCI Risk Factors Cervical Cancer (858 patients, 36 features)
  Target: Biopsy (primary) | also: Hinselmann, Schiller, Citology
  Techniques: Stratified K-Fold CV, SMOTE, Random Forest
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ─────────────────────────────────────────────
# 1. LOAD & EXPLORE
# ─────────────────────────────────────────────
df = pd.read_csv('/tmp/cancer_data/cervical/risk_factors_cervical_cancer.csv')
print(f"Dataset shape: {df.shape}")
print(f"Missing values (marked as '?'): {(df == '?').sum().sum()}")

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')

TARGET = 'Biopsy'
# Features: drop all 4 diagnostic test columns to avoid leakage
DROP_COLS = ['Hinselmann', 'Schiller', 'Citology', 'Biopsy']
X = df.drop(columns=DROP_COLS)
y = df[TARGET]

print(f"\nTarget distribution:\n{y.value_counts()}")
print(f"Class imbalance ratio: {y.value_counts()[0]/y.value_counts()[1]:.1f}:1")

# ─────────────────────────────────────────────
# 2. PIPELINE (Impute → Scale → SMOTE → RF)
# ─────────────────────────────────────────────
clf = ImbPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42, k_neighbors=3)),
    ('model', RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight='balanced',
        random_state=42, n_jobs=-1))
])

# ─────────────────────────────────────────────
# 3. STRATIFIED K-FOLD CROSS-VALIDATION (k=10)
# ─────────────────────────────────────────────
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

cv_results = cross_validate(
    clf, X, y, cv=skf,
    scoring=['roc_auc', 'f1', 'precision', 'recall'],
    return_train_score=True
)

print("\n" + "="*50)
print("STRATIFIED 10-FOLD CROSS-VALIDATION RESULTS")
print("="*50)
for metric in ['roc_auc', 'f1', 'precision', 'recall']:
    scores = cv_results[f'test_{metric}']
    print(f"{metric.upper():12s}: {scores.mean():.4f} ± {scores.std():.4f}")

# ─────────────────────────────────────────────
# 4. COLLECT OOF PREDICTIONS FOR PLOTTING
# ─────────────────────────────────────────────
oof_probs = np.zeros(len(y))
oof_preds = np.zeros(len(y))
conf_matrices = []

X_arr = X.values
y_arr = y.values

for train_idx, val_idx in skf.split(X_arr, y_arr):
    X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
    y_tr, y_val = y_arr[train_idx], y_arr[val_idx]
    clf.fit(X_tr, y_tr)
    oof_probs[val_idx] = clf.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = clf.predict(X_val)
    conf_matrices.append(confusion_matrix(y_val, clf.predict(X_val)))

# Aggregate confusion matrix
agg_cm = sum(conf_matrices)

# Feature importance (fit on full data)
imputer = SimpleImputer(strategy='median')
X_imp = imputer.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)
sm = SMOTE(random_state=42, k_neighbors=3)
X_res, y_res = sm.fit_resample(X_scaled, y)
rf_full = RandomForestClassifier(n_estimators=200, max_depth=8,
                                  class_weight='balanced', random_state=42)
rf_full.fit(X_res, y_res)
feat_imp = pd.Series(rf_full.feature_importances_, index=X.columns).sort_values(ascending=False)

# ─────────────────────────────────────────────
# 5. VISUALIZATIONS
# ─────────────────────────────────────────────
palette = {'pos': '#C0392B', 'neg': '#2980B9', 'bg': '#F8F9FA',
           'grid': '#ECEDEE', 'accent': '#8E44AD'}

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#FFFFFF')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

# ── A) ROC Curve (OOF) ──
ax1 = fig.add_subplot(gs[0, 0])
fpr, tpr, _ = roc_curve(y_arr, oof_probs)
auc = roc_auc_score(y_arr, oof_probs)
ax1.fill_between(fpr, tpr, alpha=0.15, color=palette['pos'])
ax1.plot(fpr, tpr, color=palette['pos'], lw=2.5, label=f'ROC (AUC = {auc:.3f})')
ax1.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax1.set_xlabel('False Positive Rate', fontsize=11)
ax1.set_ylabel('True Positive Rate', fontsize=11)
ax1.set_title('ROC Curve (Out-of-Fold)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.set_facecolor(palette['bg'])
ax1.grid(True, color=palette['grid'])

# ── B) Confusion Matrix ──
ax2 = fig.add_subplot(gs[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=agg_cm, display_labels=['No Cancer', 'Cancer'])
disp.plot(ax=ax2, colorbar=False, cmap='Reds')
ax2.set_title('Aggregated Confusion Matrix\n(10-Fold OOF)', fontsize=13, fontweight='bold')

# ── C) Feature Importance (Top 15) ──
ax3 = fig.add_subplot(gs[0, 2])
top15 = feat_imp.head(15)
colors = [palette['pos'] if i < 5 else palette['neg'] for i in range(15)]
ax3.barh(top15.index[::-1], top15.values[::-1], color=colors[::-1], edgecolor='white')
ax3.set_xlabel('Importance Score', fontsize=11)
ax3.set_title('Top 15 Feature Importances\n(Random Forest)', fontsize=13, fontweight='bold')
ax3.set_facecolor(palette['bg'])
ax3.grid(True, axis='x', color=palette['grid'])

# ── D) CV Score Distribution ──
ax4 = fig.add_subplot(gs[1, 0])
metrics_plot = {
    'ROC-AUC': cv_results['test_roc_auc'],
    'F1': cv_results['test_f1'],
    'Precision': cv_results['test_precision'],
    'Recall': cv_results['test_recall'],
}
bp = ax4.boxplot(metrics_plot.values(), labels=metrics_plot.keys(),
                 patch_artist=True, notch=False,
                 medianprops={'color': 'white', 'linewidth': 2})
colors_box = [palette['pos'], palette['neg'], palette['accent'], '#27AE60']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax4.set_ylabel('Score', fontsize=11)
ax4.set_title('CV Metric Distribution\n(10-Fold)', fontsize=13, fontweight='bold')
ax4.set_facecolor(palette['bg'])
ax4.grid(True, axis='y', color=palette['grid'])
ax4.set_ylim(0, 1.1)

# ── E) Class Imbalance ──
ax5 = fig.add_subplot(gs[1, 1])
class_counts = y.value_counts()
bars = ax5.bar(['No Cancer (0)', 'Cancer (1)'], class_counts.values,
               color=[palette['neg'], palette['pos']], edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, class_counts.values):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
             f'n={val}', ha='center', fontsize=11, fontweight='bold')
ax5.set_ylabel('Count', fontsize=11)
ax5.set_title('Class Distribution\n(SMOTE Applied for Training)', fontsize=13, fontweight='bold')
ax5.set_facecolor(palette['bg'])
ax5.grid(True, axis='y', color=palette['grid'])

# ── F) Score table ──
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
table_data = []
for metric in ['roc_auc', 'f1', 'precision', 'recall']:
    scores = cv_results[f'test_{metric}']
    table_data.append([metric.upper(), f"{scores.mean():.4f}", f"{scores.std():.4f}",
                       f"{scores.min():.4f}", f"{scores.max():.4f}"])
table = ax6.table(cellText=table_data,
                  colLabels=['Metric', 'Mean', 'Std', 'Min', 'Max'],
                  cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2.0)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor(palette['pos'])
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#FFF5F5')
    cell.set_edgecolor('#DDDDDD')
ax6.set_title('Cross-Validation Summary', fontsize=13, fontweight='bold', pad=60)

fig.suptitle('Cervical Cancer Risk Classification Pipeline\n'
             'Random Forest + SMOTE + Stratified 10-Fold CV  |  858 Patients, 32 Features',
             fontsize=15, fontweight='bold', y=0.98)

plt.savefig('/mnt/user-data/outputs/cervical_cancer_pipeline.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("\n✅ Saved: cervical_cancer_pipeline.png")
print(f"   OOF AUC: {auc:.4f}")
