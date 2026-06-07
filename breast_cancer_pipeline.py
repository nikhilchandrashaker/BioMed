"""
=============================================================
  BREAST CANCER RECURRENCE CLASSIFICATION PIPELINE
  Dataset: UCI Breast Cancer Ljubljana (286 instances, 9 features)
  Target: no-recurrence-events vs recurrence-events
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

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, cross_validate, cross_val_score
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
from sklearn.impute import SimpleImputer

# ─────────────────────────────────────────────
# 1. LOAD & ENCODE
# ─────────────────────────────────────────────
COLS = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes',
        'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']

df = pd.read_csv('/tmp/cancer_data/breast/breast-cancer.data',
                 header=None, names=COLS, na_values='?')

print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"\nTarget distribution:\n{df['Class'].value_counts()}")

# Encode target
le = LabelEncoder()
y = le.fit_transform(df['Class'])  # 0=no-recurrence, 1=recurrence
print(f"\nEncoded classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")

X = df.drop(columns=['Class'])

# Ordinal mappings for clinical variables
age_cats = ['10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90-99']
size_cats = ['0-4','5-9','10-14','15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59']
nodes_cats = ['0-2','3-5','6-8','9-11','12-14','15-17','18-20','21-23','24-26','27-29','30-32','33-35','36-39']

# Encode each column
def encode_ordinal(series, cats):
    mapping = {v: i for i, v in enumerate(cats)}
    return series.map(mapping)

X = X.copy()
X['age'] = encode_ordinal(X['age'], age_cats)
X['tumor-size'] = encode_ordinal(X['tumor-size'], size_cats)
X['inv-nodes'] = encode_ordinal(X['inv-nodes'], nodes_cats)
X['menopause'] = X['menopause'].map({'lt40': 0, 'premeno': 1, 'ge40': 2})
X['node-caps'] = X['node-caps'].map({'no': 0, 'yes': 1})
X['breast'] = X['breast'].map({'left': 0, 'right': 1})
X['breast-quad'] = X['breast-quad'].map({
    'left_low': 0, 'left_up': 1, 'right_low': 2, 'right_up': 3,
    'central': 4, 'left_low ': 0})
X['irradiat'] = X['irradiat'].map({'no': 0, 'yes': 1})
X['deg-malig'] = X['deg-malig'].astype(float)

# ─────────────────────────────────────────────
# 2. PIPELINE
# ─────────────────────────────────────────────
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
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
# For LOO, collect probabilities manually
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
palette = {'pos': '#8E44AD', 'neg': '#2980B9', 'bg': '#F8F9FA',
           'grid': '#ECEDEE', 'accent': '#16A085'}

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
                               display_labels=['No Recurrence', 'Recurrence'])
disp.plot(ax=ax2, colorbar=False, cmap='Purples')
ax2.set_title('Aggregated Confusion Matrix\n(10-Fold OOF)', fontsize=13, fontweight='bold')

# ── C) Feature Importance ──
ax3 = fig.add_subplot(gs[0, 2])
colors_feat = [palette['pos'] if i < 3 else palette['neg'] for i in range(len(feat_imp))]
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
ax4.axhline(fold_aucs.mean(), color='red', linestyle='--', lw=2,
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
bars = ax5.bar(['No Recurrence', 'Recurrence'],
               [class_counts[0], class_counts[1]],
               color=[palette['neg'], palette['pos']], edgecolor='white')
for bar, val in zip(bars, [class_counts[0], class_counts[1]]):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'n={val}', ha='center', fontsize=12, fontweight='bold')
ax5.set_ylabel('Count', fontsize=11)
ax5.set_title('Class Distribution', fontsize=13, fontweight='bold')
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
        cell.set_facecolor('#F5EEF8')
    cell.set_edgecolor('#DDDDDD')
ax6.set_title('Validation Summary\nSKF vs LOO', fontsize=13, fontweight='bold', pad=60)

fig.suptitle('Breast Cancer Recurrence Classification Pipeline\n'
             'Gradient Boosting + Stratified K-Fold + Leave-One-Out CV  |  286 Patients, 9 Features',
             fontsize=15, fontweight='bold', y=0.98)

plt.savefig('/mnt/user-data/outputs/breast_cancer_pipeline.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("\n✅ Saved: breast_cancer_pipeline.png")
print(f"   SKF AUC: {auc_skf:.4f} | LOO AUC: {loo_auc:.4f}")
