"""
=============================================================
  HEPATITIS SURVIVAL CLASSIFICATION PIPELINE
  Dataset: UCI Hepatitis Domain (155 instances, 19 features)
  Target: LIVE vs DIE
  Techniques: Stratified K-Fold CV, Leave-One-Out CV, Gradient Boosting
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

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, cross_validate
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.impute import SimpleImputer

# ─────────────────────────────────────────────
# 1. LOAD & ENCODE
# ─────────────────────────────────────────────
COLS = ['Class', 'age', 'sex', 'steroid', 'antivirals', 'fatigue', 'malaise',
        'anorexia', 'liver-big', 'liver-firm', 'spleen-palpable', 'spiders',
        'ascites', 'varices', 'bilirubin', 'alk-phosphate', 'sgot',
        'albumin', 'protime', 'histology']

df = pd.read_csv('/tmp/hepatitis_data/hepatitis.data',
                 header=None, names=COLS, na_values='?')

print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")

# Original coding: Class 1=DIE, 2=LIVE
class_map = {1: 'DIE', 2: 'LIVE'}
print(f"\nTarget distribution:\n{df['Class'].map(class_map).value_counts()}")

# Target: predict DIE as the positive (clinically critical) class
y = (df['Class'] == 1).astype(int).values  # 1 = DIE, 0 = LIVE
X = df.drop(columns=['Class']).copy()

# Binary attributes are coded 1=no, 2=yes in the raw file -> rescale to 0/1
binary_cols = ['steroid', 'antivirals', 'fatigue', 'malaise', 'anorexia',
               'liver-big', 'liver-firm', 'spleen-palpable', 'spiders',
               'ascites', 'varices', 'histology']
for c in binary_cols:
    X[c] = X[c].map({1: 0, 2: 1})

# sex is coded 1=male, 2=female -> rescale to 0/1
X['sex'] = X['sex'].map({1: 0, 2: 1})

# Continuous clinical labs stay numeric as-is
continuous_cols = ['age', 'bilirubin', 'alk-phosphate', 'sgot', 'albumin', 'protime']
for c in continuous_cols:
    X[c] = X[c].astype(float)

print(f"\nClinical features: {list(X.columns)}")

# ─────────────────────────────────────────────
# 2. PIPELINE
# ─────────────────────────────────────────────
# Median imputation suits this mixed continuous/binary clinical panel,
# since several labs (protime, albumin, alk-phosphate) have >10% missingness.
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        subsample=0.8, random_state=42))
])

# ─────────────────────────────────────────────
# 3. STRATIFIED K-FOLD (k=10)
# ─────────────────────────────────────────────
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_results = cross_validate(
    pipe, X, y, cv=skf,
    scoring=['roc_auc', 'f1', 'precision', 'recall'],
    return_train_score=True
)

print("\n" + "="*50)
print("STRATIFIED 10-FOLD CV RESULTS")
print("="*50)
for metric in ['roc_auc', 'f1', 'precision', 'recall']:
    scores = cv_results[f'test_{metric}']
    print(f"{metric.upper():12s}: {scores.mean():.4f} ± {scores.std():.4f}")

# ─────────────────────────────────────────────
# 4. LEAVE-ONE-OUT CV
# ─────────────────────────────────────────────
loo = LeaveOneOut()
X_arr, y_arr = X.values, y

loo_probs = np.zeros(len(y_arr))
loo_preds = np.zeros(len(y_arr))

for train_idx, test_idx in loo.split(X_arr):
    pipe.fit(X_arr[train_idx], y_arr[train_idx])
    loo_probs[test_idx] = pipe.predict_proba(X_arr[test_idx])[:, 1]
    loo_preds[test_idx] = pipe.predict(X_arr[test_idx])

loo_auc = roc_auc_score(y_arr, loo_probs)
loo_acc = (loo_preds == y_arr).mean()
print(f"\nLEAVE-ONE-OUT CV:")
print(f"  AUC  : {loo_auc:.4f}")
print(f"  Acc  : {loo_acc:.4f}")

# OOF skf predictions
oof_probs = np.zeros(len(y_arr))
oof_preds = np.zeros(len(y_arr))
conf_mats = []
for train_idx, val_idx in skf.split(X_arr, y_arr):
    pipe.fit(X_arr[train_idx], y_arr[train_idx])
    oof_probs[val_idx] = pipe.predict_proba(X_arr[val_idx])[:, 1]
    oof_preds[val_idx] = pipe.predict(X_arr[val_idx])
    conf_mats.append(confusion_matrix(y_arr[val_idx], pipe.predict(X_arr[val_idx])))

agg_cm = sum(conf_mats)

# Feature importance
pipe.fit(X_arr, y_arr)
feat_imp = pd.Series(pipe.named_steps['model'].feature_importances_, index=X.columns).sort_values(ascending=False)

# ─────────────────────────────────────────────
# 5. VISUALIZATIONS
# ─────────────────────────────────────────────
palette = {'pos': '#C0392B', 'neg': '#27AE60', 'bg': '#F8F9FA',
           'grid': '#ECEDEE', 'accent': '#2980B9'}

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#FFFFFF')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

# ── A) ROC Curve - SKF + LOO ──
ax1 = fig.add_subplot(gs[0, 0])
fpr_skf, tpr_skf, _ = roc_curve(y_arr, oof_probs)
fpr_loo, tpr_loo, _ = roc_curve(y_arr, loo_probs)
auc_skf = roc_auc_score(y_arr, oof_probs)

ax1.fill_between(fpr_skf, tpr_skf, alpha=0.12, color=palette['pos'])
ax1.plot(fpr_skf, tpr_skf, color=palette['pos'], lw=2.5,
         label=f'10-Fold SKF (AUC={auc_skf:.3f})')
ax1.plot(fpr_loo, tpr_loo, color=palette['accent'], lw=2, linestyle='--',
         label=f'LOO (AUC={loo_auc:.3f})')
ax1.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax1.set_xlabel('False Positive Rate', fontsize=11)
ax1.set_ylabel('True Positive Rate', fontsize=11)
ax1.set_title('ROC Curves\nStratified K-Fold vs LOO', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.set_facecolor(palette['bg'])
ax1.grid(True, color=palette['grid'])

# ── B) Confusion Matrix ──
ax2 = fig.add_subplot(gs[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=agg_cm,
                               display_labels=['LIVE', 'DIE'])
disp.plot(ax=ax2, colorbar=False, cmap='Reds')
ax2.set_title('Aggregated Confusion Matrix\n(10-Fold OOF)', fontsize=13, fontweight='bold')

# ── C) Feature Importance ──
ax3 = fig.add_subplot(gs[0, 2])
colors_feat = [palette['pos'] if i < 3 else palette['accent'] for i in range(len(feat_imp))]
ax3.barh(feat_imp.index[::-1], feat_imp.values[::-1],
         color=colors_feat[::-1], edgecolor='white')
ax3.set_xlabel('Importance Score', fontsize=11)
ax3.set_title('Feature Importances\n(Gradient Boosting)', fontsize=13, fontweight='bold')
ax3.set_facecolor(palette['bg'])
ax3.grid(True, axis='x', color=palette['grid'])

# ── D) CV Fold AUC ──
ax4 = fig.add_subplot(gs[1, 0])
fold_aucs = cv_results['test_roc_auc']
fold_nums = np.arange(1, len(fold_aucs)+1)
ax4.bar(fold_nums, fold_aucs, color=[palette['pos'] if a >= fold_aucs.mean()
        else palette['neg'] for a in fold_aucs], edgecolor='white', alpha=0.85)
ax4.axhline(fold_aucs.mean(), color='black', linestyle='--', lw=2,
            label=f'Mean AUC = {fold_aucs.mean():.3f}')
ax4.set_xlabel('Fold', fontsize=11)
ax4.set_ylabel('ROC-AUC', fontsize=11)
ax4.set_title('AUC per CV Fold\n(Stratified 10-Fold)', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.set_facecolor(palette['bg'])
ax4.grid(True, axis='y', color=palette['grid'])
ax4.set_xticks(fold_nums)
ax4.set_ylim(0, 1.1)

# ── E) Class Distribution ──
ax5 = fig.add_subplot(gs[1, 1])
class_counts = pd.Series(y_arr).value_counts()
bars = ax5.bar(['LIVE', 'DIE'],
               [class_counts[0], class_counts[1]],
               color=[palette['neg'], palette['pos']], edgecolor='white')
for bar, val in zip(bars, [class_counts[0], class_counts[1]]):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'n={val}', ha='center', fontsize=12, fontweight='bold')
ax5.set_ylabel('Count', fontsize=11)
ax5.set_title('Class Distribution\n(155 patients)', fontsize=13, fontweight='bold')
ax5.set_facecolor(palette['bg'])
ax5.grid(True, axis='y', color=palette['grid'])

# ── F) LOO vs SKF comparison table ──
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
table_data = [
    ['Method', 'ROC-AUC', 'Accuracy'],
    ['Strat. K-Fold (10)', f"{cv_results['test_roc_auc'].mean():.4f}",
     f"{(oof_preds==y_arr).mean():.4f}"],
    ['Leave-One-Out', f"{loo_auc:.4f}", f"{loo_acc:.4f}"],
    ['', '', ''],
    ['Metric (SKF)', 'Mean', '± Std'],
]
for metric in ['roc_auc', 'f1', 'precision', 'recall']:
    scores = cv_results[f'test_{metric}']
    table_data.append([metric.upper(), f"{scores.mean():.4f}", f"±{scores.std():.4f}"])

table = ax6.table(cellText=table_data[1:],
                  colLabels=table_data[0],
                  cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1.2, 2.0)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor(palette['pos'])
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#FDEDEC')
    cell.set_edgecolor('#DDDDDD')
ax6.set_title('Validation Summary\nSKF vs LOO', fontsize=13, fontweight='bold', pad=60)

fig.suptitle('Hepatitis Survival Classification Pipeline\n'
             'Gradient Boosting + Stratified K-Fold + Leave-One-Out CV  |  155 Patients, 19 Features',
             fontsize=15, fontweight='bold', y=0.98)

plt.savefig('/mnt/user-data/outputs/hepatitis_pipeline.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("\n✅ Saved: hepatitis_pipeline.png")
print(f"   SKF AUC: {auc_skf:.4f} | LOO AUC: {loo_auc:.4f}")
