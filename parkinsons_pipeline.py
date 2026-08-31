"""
=============================================================
  PARKINSON'S DISEASE VOICE CLASSIFICATION PIPELINE
  Dataset: UCI Oxford Parkinson's Disease Detection (195 recordings, 22 features)
  Target: healthy vs Parkinson's (status)
  Techniques: Stratified K-Fold CV, Subject-Grouped K-Fold CV,
              Leave-One-Out CV, Gradient Boosting
=============================================================
  NOTE: There are ~6 voice recordings per subject. Ordinary
  Stratified K-Fold can leak the same subject's voice into both
  train and test folds and inflate scores. Subject-Grouped K-Fold
  (GroupKFold on the patient id parsed from `name`) gives the more
  trustworthy, leak-free estimate for how the model performs on a
  genuinely unseen patient.
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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (StratifiedKFold, GroupKFold, LeaveOneOut,
                                     cross_validate)
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             ConfusionMatrixDisplay)

# ─────────────────────────────────────────────
# 1. LOAD & PREP
# ─────────────────────────────────────────────
df = pd.read_csv('/tmp/parkinsons_data/parkinsons.data')

print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"\nTarget distribution (0=healthy, 1=Parkinson's):\n{df['status'].value_counts()}")

# Parse subject id out of "name", e.g. phon_R01_S01_1 -> S01
df['subject'] = df['name'].str.extract(r'(S\d+)')
n_subjects = df['subject'].nunique()
print(f"\nUnique subjects: {n_subjects}  |  Recordings: {len(df)}")

y = df['status'].values
groups = df['subject'].values
X = df.drop(columns=['name', 'status', 'subject'])
print(f"\nVoice measures: {list(X.columns)}")

# ─────────────────────────────────────────────
# 2. PIPELINE
# ─────────────────────────────────────────────
# Voice-signal features span very different scales (Hz vs. jitter %
# vs. nonlinear complexity measures), so standardize before boosting.
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        subsample=0.8, random_state=42))
])

X_arr, y_arr = X.values, y

# ─────────────────────────────────────────────
# 3. STRATIFIED K-FOLD (k=10) — naive, recording-level
# ─────────────────────────────────────────────
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_results = cross_validate(
    pipe, X, y, cv=skf,
    scoring=['roc_auc', 'f1', 'precision', 'recall'],
    return_train_score=True
)

print("\n" + "="*50)
print("STRATIFIED 10-FOLD CV RESULTS (recording-level, optimistic)")
print("="*50)
for metric in ['roc_auc', 'f1', 'precision', 'recall']:
    scores = cv_results[f'test_{metric}']
    print(f"{metric.upper():12s}: {scores.mean():.4f} ± {scores.std():.4f}")

oof_probs = np.zeros(len(y_arr))
oof_preds = np.zeros(len(y_arr))
conf_mats = []
for train_idx, val_idx in skf.split(X_arr, y_arr):
    pipe.fit(X_arr[train_idx], y_arr[train_idx])
    oof_probs[val_idx] = pipe.predict_proba(X_arr[val_idx])[:, 1]
    oof_preds[val_idx] = pipe.predict(X_arr[val_idx])
    conf_mats.append(confusion_matrix(y_arr[val_idx], pipe.predict(X_arr[val_idx])))
agg_cm = sum(conf_mats)
auc_skf = roc_auc_score(y_arr, oof_probs)

# ─────────────────────────────────────────────
# 4. SUBJECT-GROUPED K-FOLD — leak-free, patient-level
# ─────────────────────────────────────────────
n_groups = min(10, n_subjects)
gkf = GroupKFold(n_splits=n_groups)
gkf_probs = np.zeros(len(y_arr))
gkf_preds = np.zeros(len(y_arr))
for train_idx, val_idx in gkf.split(X_arr, y_arr, groups=groups):
    pipe.fit(X_arr[train_idx], y_arr[train_idx])
    gkf_probs[val_idx] = pipe.predict_proba(X_arr[val_idx])[:, 1]
    gkf_preds[val_idx] = pipe.predict(X_arr[val_idx])

gkf_auc = roc_auc_score(y_arr, gkf_probs)
gkf_acc = (gkf_preds == y_arr).mean()
print(f"\nSUBJECT-GROUPED {n_groups}-FOLD CV (leak-free, patient-level):")
print(f"  AUC  : {gkf_auc:.4f}")
print(f"  Acc  : {gkf_acc:.4f}")
print(f"  (compare to naive SKF AUC: {auc_skf:.4f} — the gap is the leakage effect)")

# ─────────────────────────────────────────────
# 5. LEAVE-ONE-OUT CV (recording-level, for completeness)
# ─────────────────────────────────────────────
loo = LeaveOneOut()
loo_probs = np.zeros(len(y_arr))
loo_preds = np.zeros(len(y_arr))
for train_idx, test_idx in loo.split(X_arr):
    pipe.fit(X_arr[train_idx], y_arr[train_idx])
    loo_probs[test_idx] = pipe.predict_proba(X_arr[test_idx])[:, 1]
    loo_preds[test_idx] = pipe.predict(X_arr[test_idx])
loo_auc = roc_auc_score(y_arr, loo_probs)
loo_acc = (loo_preds == y_arr).mean()
print(f"\nLEAVE-ONE-OUT CV (recording-level):")
print(f"  AUC  : {loo_auc:.4f}")
print(f"  Acc  : {loo_acc:.4f}")

# Feature importance (fit on all data)
pipe.fit(X_arr, y_arr)
feat_imp = pd.Series(pipe.named_steps['model'].feature_importances_, index=X.columns).sort_values(ascending=False)

# ─────────────────────────────────────────────
# 6. VISUALIZATIONS
# ─────────────────────────────────────────────
palette = {'pos': '#8E44AD', 'neg': '#16A085', 'bg': '#F8F9FA',
           'grid': '#ECEDEE', 'accent': '#D35400'}

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#FFFFFF')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

# ── A) ROC Curve - SKF vs GroupKFold vs LOO ──
ax1 = fig.add_subplot(gs[0, 0])
fpr_skf, tpr_skf, _ = roc_curve(y_arr, oof_probs)
fpr_gkf, tpr_gkf, _ = roc_curve(y_arr, gkf_probs)
fpr_loo, tpr_loo, _ = roc_curve(y_arr, loo_probs)

ax1.fill_between(fpr_gkf, tpr_gkf, alpha=0.12, color=palette['accent'])
ax1.plot(fpr_skf, tpr_skf, color=palette['pos'], lw=2, linestyle=':',
         label=f'Naive SKF (AUC={auc_skf:.3f})')
ax1.plot(fpr_gkf, tpr_gkf, color=palette['accent'], lw=2.5,
         label=f'Grouped K-Fold (AUC={gkf_auc:.3f})')
ax1.plot(fpr_loo, tpr_loo, color=palette['neg'], lw=2, linestyle='--',
         label=f'LOO (AUC={loo_auc:.3f})')
ax1.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax1.set_xlabel('False Positive Rate', fontsize=11)
ax1.set_ylabel('True Positive Rate', fontsize=11)
ax1.set_title('ROC Curves\nNaive vs Subject-Grouped vs LOO', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.set_facecolor(palette['bg'])
ax1.grid(True, color=palette['grid'])

# ── B) Confusion Matrix (subject-grouped, most trustworthy) ──
ax2 = fig.add_subplot(gs[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_arr, gkf_preds),
                               display_labels=['Healthy', "Parkinson's"])
disp.plot(ax=ax2, colorbar=False, cmap='Purples')
ax2.set_title("Confusion Matrix\n(Subject-Grouped OOF)", fontsize=13, fontweight='bold')

# ── C) Feature Importance ──
ax3 = fig.add_subplot(gs[0, 2])
top_feat = feat_imp.head(15)
colors_feat = [palette['pos'] if i < 3 else palette['neg'] for i in range(len(top_feat))]
ax3.barh(top_feat.index[::-1], top_feat.values[::-1],
         color=colors_feat[::-1], edgecolor='white')
ax3.set_xlabel('Importance Score', fontsize=11)
ax3.set_title('Top 15 Feature Importances\n(Gradient Boosting)', fontsize=13, fontweight='bold')
ax3.set_facecolor(palette['bg'])
ax3.grid(True, axis='x', color=palette['grid'])

# ── D) CV Fold AUC (subject-grouped) ──
ax4 = fig.add_subplot(gs[1, 0])
fold_aucs = []
for train_idx, val_idx in gkf.split(X_arr, y_arr, groups=groups):
    pipe.fit(X_arr[train_idx], y_arr[train_idx])
    p = pipe.predict_proba(X_arr[val_idx])[:, 1]
    if len(np.unique(y_arr[val_idx])) > 1:
        fold_aucs.append(roc_auc_score(y_arr[val_idx], p))
fold_aucs = np.array(fold_aucs)
fold_nums = np.arange(1, len(fold_aucs)+1)
ax4.bar(fold_nums, fold_aucs, color=[palette['accent'] if a >= fold_aucs.mean()
        else palette['neg'] for a in fold_aucs], edgecolor='white', alpha=0.85)
ax4.axhline(fold_aucs.mean(), color='black', linestyle='--', lw=2,
            label=f'Mean AUC = {fold_aucs.mean():.3f}')
ax4.set_xlabel('Fold', fontsize=11)
ax4.set_ylabel('ROC-AUC', fontsize=11)
ax4.set_title('AUC per Fold\n(Subject-Grouped K-Fold)', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.set_facecolor(palette['bg'])
ax4.grid(True, axis='y', color=palette['grid'])
ax4.set_xticks(fold_nums)
ax4.set_ylim(0, 1.1)

# ── E) Class Distribution ──
ax5 = fig.add_subplot(gs[1, 1])
class_counts = pd.Series(y_arr).value_counts()
bars = ax5.bar(['Healthy', "Parkinson's"],
               [class_counts[0], class_counts[1]],
               color=[palette['neg'], palette['pos']], edgecolor='white')
for bar, val in zip(bars, [class_counts[0], class_counts[1]]):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'n={val}', ha='center', fontsize=12, fontweight='bold')
ax5.set_ylabel('Count (recordings)', fontsize=11)
ax5.set_title(f'Class Distribution\n({len(y_arr)} recordings, {n_subjects} subjects)', fontsize=13, fontweight='bold')
ax5.set_facecolor(palette['bg'])
ax5.grid(True, axis='y', color=palette['grid'])

# ── F) Validation method comparison table ──
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
table_data = [
    ['Method', 'ROC-AUC', 'Accuracy'],
    ['Naive Strat. K-Fold', f"{auc_skf:.4f}", f"{(oof_preds==y_arr).mean():.4f}"],
    ['Subject-Grouped K-Fold', f"{gkf_auc:.4f}", f"{gkf_acc:.4f}"],
    ['Leave-One-Out', f"{loo_auc:.4f}", f"{loo_acc:.4f}"],
    ['', '', ''],
    ['Metric (Naive SKF)', 'Mean', '± Std'],
]
for metric in ['roc_auc', 'f1', 'precision', 'recall']:
    scores = cv_results[f'test_{metric}']
    table_data.append([metric.upper(), f"{scores.mean():.4f}", f"±{scores.std():.4f}"])

table = ax6.table(cellText=table_data[1:],
                  colLabels=table_data[0],
                  cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.15, 1.9)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor(palette['pos'])
        cell.set_text_props(color='white', fontweight='bold')
    elif r == 2:  # subject-grouped row: the trustworthy estimate
        cell.set_facecolor('#FDEBD0')
    elif r % 2 == 0:
        cell.set_facecolor('#F5EEF8')
    cell.set_edgecolor('#DDDDDD')
ax6.set_title('Validation Summary\nNaive vs Leak-Free Estimates', fontsize=13, fontweight='bold', pad=60)

fig.suptitle("Parkinson's Disease Voice Classification Pipeline\n"
             f'Gradient Boosting + Subject-Grouped K-Fold + LOO CV  |  {len(y_arr)} Recordings, {n_subjects} Subjects, {X.shape[1]} Voice Features',
             fontsize=15, fontweight='bold', y=0.98)

plt.savefig('/mnt/user-data/outputs/parkinsons_pipeline.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("\n✅ Saved: parkinsons_pipeline.png")
print(f"   Naive SKF AUC: {auc_skf:.4f} | Subject-Grouped AUC: {gkf_auc:.4f} | LOO AUC: {loo_auc:.4f}")
