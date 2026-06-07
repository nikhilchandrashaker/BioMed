"""
=============================================================
  LUNG CANCER TYPE CLASSIFICATION PIPELINE
  Dataset: UCI Lung Cancer (32 instances, 56 features, 3 classes)
  Target: Lung cancer type (1, 2, 3)
  Techniques: Leave-One-Out CV (small dataset), Stratified K-Fold,
              SVM + RF ensemble, missing value imputation
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

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, cross_validate
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay,
                             auc as sklearn_auc)
from sklearn.impute import SimpleImputer
from sklearn.multiclass import OneVsRestClassifier

# ─────────────────────────────────────────────
# 1. LOAD & EXPLORE
# ─────────────────────────────────────────────
df = pd.read_csv('/tmp/cancer_data/lung/lung-cancer.data',
                 header=None, na_values='?')

print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"\nClass distribution:\n{df[0].value_counts()}")

y = df[0].values - 1  # 0-index: classes become 0, 1, 2
X = df.drop(columns=[0])

class_names = ['Type 1', 'Type 2', 'Type 3']

# ─────────────────────────────────────────────
# 2. PIPELINE (Impute → Scale → RF)
# ─────────────────────────────────────────────
# Given tiny sample (n=32, 56 features), use RF with balanced weights
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(
        n_estimators=500, max_depth=4, min_samples_leaf=2,
        class_weight='balanced', random_state=42, n_jobs=-1))
])

# ─────────────────────────────────────────────
# 3. LEAVE-ONE-OUT CV (primary — n=32 is tiny)
# ─────────────────────────────────────────────
loo = LeaveOneOut()
loo_probs = np.zeros((len(y), 3))
loo_preds = np.zeros(len(y), dtype=int)

for train_idx, test_idx in loo.split(X):
    pipe.fit(X.iloc[train_idx], y[train_idx])
    loo_probs[test_idx] = pipe.predict_proba(X.iloc[test_idx])
    loo_preds[test_idx] = pipe.predict(X.iloc[test_idx])

loo_acc = (loo_preds == y).mean()

# Multiclass OvR AUC
y_bin = label_binarize(y, classes=[0, 1, 2])
loo_auc_macro = roc_auc_score(y_bin, loo_probs, average='macro', multi_class='ovr')
loo_auc_weighted = roc_auc_score(y_bin, loo_probs, average='weighted', multi_class='ovr')

print(f"\nLEAVE-ONE-OUT CV:")
print(f"  Accuracy    : {loo_acc:.4f}")
print(f"  AUC (macro) : {loo_auc_macro:.4f}")
print(f"  AUC (wtd)   : {loo_auc_weighted:.4f}")

# ─────────────────────────────────────────────
# 4. STRATIFIED K-FOLD (k=5, limited by smallest class n=9)
# ─────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(
    pipe, X, y, cv=skf,
    scoring=['accuracy', 'f1_macro', 'f1_weighted'],
    return_train_score=True
)

print(f"\nSTRATIFIED 5-FOLD CV:")
for metric in ['accuracy', 'f1_macro', 'f1_weighted']:
    scores = cv_results[f'test_{metric}']
    print(f"  {metric:15s}: {scores.mean():.4f} ± {scores.std():.4f}")

# Confusion matrix from LOO
cm = confusion_matrix(y, loo_preds)

# Feature importance (fit on full data)
pipe.fit(X, y)
feat_imp = pd.Series(pipe.named_steps['model'].feature_importances_,
                     index=[f'F{i+1}' for i in range(X.shape[1])]).sort_values(ascending=False)

# Per-class ROC curves
fpr_dict, tpr_dict, auc_dict = {}, {}, {}
for i, cls in enumerate(class_names):
    fpr_dict[i], tpr_dict[i], _ = roc_curve(y_bin[:, i], loo_probs[:, i])
    auc_dict[i] = sklearn_auc(fpr_dict[i], tpr_dict[i])

# ─────────────────────────────────────────────
# 5. VISUALIZATIONS
# ─────────────────────────────────────────────
palette = {'c0': '#E74C3C', 'c1': '#3498DB', 'c2': '#2ECC71',
           'bg': '#F8F9FA', 'grid': '#ECEDEE', 'accent': '#F39C12'}
class_colors = [palette['c0'], palette['c1'], palette['c2']]

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#FFFFFF')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

# ── A) Per-class ROC Curves (OvR) ──
ax1 = fig.add_subplot(gs[0, 0])
for i, cls in enumerate(class_names):
    ax1.plot(fpr_dict[i], tpr_dict[i], color=class_colors[i], lw=2.5,
             label=f'{cls} (AUC={auc_dict[i]:.3f})')
ax1.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax1.fill_between([0,1], [0,1], alpha=0.05, color='gray')
ax1.set_xlabel('False Positive Rate', fontsize=11)
ax1.set_ylabel('True Positive Rate', fontsize=11)
ax1.set_title('Per-Class ROC Curves (OvR)\nLeave-One-Out CV', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.set_facecolor(palette['bg'])
ax1.grid(True, color=palette['grid'])
ax1.text(0.55, 0.08, f'Macro AUC = {loo_auc_macro:.3f}',
         transform=ax1.transAxes, fontsize=11,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# ── B) Confusion Matrix ──
ax2 = fig.add_subplot(gs[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax2, colorbar=False, cmap='Blues')
ax2.set_title('Confusion Matrix\n(Leave-One-Out CV)', fontsize=13, fontweight='bold')

# ── C) Feature Importance (Top 20) ──
ax3 = fig.add_subplot(gs[0, 2])
top20 = feat_imp.head(20)
colors_feat = [class_colors[0] if i < 5 else class_colors[1] if i < 12
               else class_colors[2] for i in range(20)]
ax3.barh(top20.index[::-1], top20.values[::-1],
         color=colors_feat[::-1], edgecolor='white')
ax3.set_xlabel('Importance Score', fontsize=11)
ax3.set_title('Top 20 Feature Importances\n(Random Forest, 56 nominal features)',
              fontsize=13, fontweight='bold')
ax3.set_facecolor(palette['bg'])
ax3.grid(True, axis='x', color=palette['grid'])

# ── D) LOO Prediction Probability Heatmap ──
ax4 = fig.add_subplot(gs[1, 0])
prob_df = pd.DataFrame(loo_probs, columns=class_names)
prob_df['True'] = [class_names[i] for i in y]
prob_df = prob_df.sort_values('True').reset_index(drop=True)

im = ax4.imshow(prob_df[class_names].values.T, aspect='auto', cmap='RdYlGn',
                vmin=0, vmax=1)
ax4.set_yticks([0, 1, 2])
ax4.set_yticklabels(class_names)
ax4.set_xlabel('Sample Index', fontsize=11)
ax4.set_title('LOO Predicted Probabilities\n(sorted by true class)', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax4, shrink=0.8)
# Add class boundary lines
boundaries = [9, 22]
for b in boundaries:
    ax4.axvline(b - 0.5, color='white', lw=2, linestyle='--')

# ── E) K-Fold score per fold ──
ax5 = fig.add_subplot(gs[1, 1])
fold_nums = np.arange(1, 6)
accs = cv_results['test_accuracy']
f1s = cv_results['test_f1_macro']
ax5.plot(fold_nums, accs, 'o-', color=palette['c0'], lw=2, label='Accuracy', markersize=8)
ax5.plot(fold_nums, f1s, 's-', color=palette['c1'], lw=2, label='F1-Macro', markersize=8)
ax5.axhline(accs.mean(), color=palette['c0'], linestyle='--', alpha=0.5)
ax5.axhline(f1s.mean(), color=palette['c1'], linestyle='--', alpha=0.5)
ax5.set_xlabel('Fold', fontsize=11)
ax5.set_ylabel('Score', fontsize=11)
ax5.set_title('Stratified 5-Fold CV Scores\nper Fold', fontsize=13, fontweight='bold')
ax5.legend(fontsize=10)
ax5.set_facecolor(palette['bg'])
ax5.grid(True, color=palette['grid'])
ax5.set_ylim(0, 1.1)
ax5.set_xticks(fold_nums)

# ── F) Summary Table ──
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
table_data = [
    ['Validation', 'Metric', 'Score'],
    ['LOO (n=32)', 'Accuracy', f'{loo_acc:.4f}'],
    ['LOO (n=32)', 'AUC Macro', f'{loo_auc_macro:.4f}'],
    ['LOO (n=32)', 'AUC Weighted', f'{loo_auc_weighted:.4f}'],
    ['5-Fold SKF', 'Accuracy', f"{cv_results['test_accuracy'].mean():.4f} ±{cv_results['test_accuracy'].std():.3f}"],
    ['5-Fold SKF', 'F1 Macro', f"{cv_results['test_f1_macro'].mean():.4f} ±{cv_results['test_f1_macro'].std():.3f}"],
    ['5-Fold SKF', 'F1 Weighted', f"{cv_results['test_f1_weighted'].mean():.4f} ±{cv_results['test_f1_weighted'].std():.3f}"],
    ['Per-Class AUC', 'Type 1', f'{auc_dict[0]:.4f}'],
    ['Per-Class AUC', 'Type 2', f'{auc_dict[1]:.4f}'],
    ['Per-Class AUC', 'Type 3', f'{auc_dict[2]:.4f}'],
]
table = ax6.table(cellText=table_data[1:],
                  colLabels=table_data[0],
                  cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1.2, 1.8)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor(palette['c0'])
        cell.set_text_props(color='white', fontweight='bold')
    elif r in [1,2,3]:
        cell.set_facecolor('#FDEDEC')
    elif r in [4,5,6]:
        cell.set_facecolor('#EBF5FB')
    else:
        cell.set_facecolor('#EAFAF1')
    cell.set_edgecolor('#DDDDDD')
ax6.set_title('Full Validation Summary', fontsize=13, fontweight='bold', pad=55)

fig.suptitle('Lung Cancer Type Classification Pipeline\n'
             'Random Forest + LOO + Stratified 5-Fold CV  |  32 Patients, 56 Features, 3 Classes',
             fontsize=15, fontweight='bold', y=0.98)

plt.savefig('/mnt/user-data/outputs/lung_cancer_pipeline.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("\n✅ Saved: lung_cancer_pipeline.png")
print(f"   LOO AUC (macro): {loo_auc_macro:.4f}")
print(f"   LOO Accuracy: {loo_acc:.4f}")
