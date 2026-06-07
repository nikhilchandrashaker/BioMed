"""
=============================================================
  HEART FAILURE SURVIVAL CLASSIFICATION PIPELINE
  Dataset: Heart Failure Clinical Records (299 instances, 12 features)
  Target: DEATH_EVENT (0=survived, 1=died)
  Techniques: Stratified K-Fold CV, Leave-One-Out CV, Gradient Boosting
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, cross_validate
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.impute import SimpleImputer

# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
df = pd.read_csv('/tmp/datasets/heart_failure/heart_failure_clinical_records_dataset.csv')
print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"\nTarget distribution:\n{df['DEATH_EVENT'].value_counts()}")

y_arr = df['DEATH_EVENT'].values
X     = df.drop(columns=['DEATH_EVENT'])
X_arr = X.values.astype(float)

# ─────────────────────────────────────────────
# 2. PIPELINE
# ─────────────────────────────────────────────
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=3, subsample=0.8, random_state=42))
])

# ─────────────────────────────────────────────
# 3. STRATIFIED 10-FOLD CV
# ─────────────────────────────────────────────
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_results = cross_validate(pipe, X_arr, y_arr, cv=skf,
    scoring=['roc_auc','f1','precision','recall'],
    return_train_score=True)

print("\n" + "="*50)
print("STRATIFIED 10-FOLD CV RESULTS")
print("="*50)
for m in ['roc_auc','f1','precision','recall']:
    s = cv_results[f'test_{m}']
    print(f"{m.upper():12s}: {s.mean():.4f} ± {s.std():.4f}")

# ─────────────────────────────────────────────
# 4. LEAVE-ONE-OUT CV
# ─────────────────────────────────────────────
loo = LeaveOneOut()
loo_probs = np.zeros(len(y_arr))
loo_preds = np.zeros(len(y_arr))
for tr, te in loo.split(X_arr):
    pipe.fit(X_arr[tr], y_arr[tr])
    loo_probs[te] = pipe.predict_proba(X_arr[te])[:, 1]
    loo_preds[te] = pipe.predict(X_arr[te])

loo_auc = roc_auc_score(y_arr, loo_probs)
loo_acc = (loo_preds == y_arr).mean()
print(f"\nLEAVE-ONE-OUT CV:  AUC={loo_auc:.4f}  Acc={loo_acc:.4f}")

# OOF SKF
oof_probs = np.zeros(len(y_arr))
oof_preds = np.zeros(len(y_arr))
conf_mats = []
for tr, va in skf.split(X_arr, y_arr):
    pipe.fit(X_arr[tr], y_arr[tr])
    oof_probs[va] = pipe.predict_proba(X_arr[va])[:, 1]
    oof_preds[va] = pipe.predict(X_arr[va])
    conf_mats.append(confusion_matrix(y_arr[va], pipe.predict(X_arr[va])))

agg_cm  = sum(conf_mats)
auc_skf = roc_auc_score(y_arr, oof_probs)

pipe.fit(X_arr, y_arr)
feat_imp = pd.Series(pipe.named_steps['model'].feature_importances_,
                     index=X.columns).sort_values(ascending=False)

# ─────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────
PAL = {'pos':'#C0392B','neg':'#2980B9','bg':'#F8F9FA','grid':'#ECEDEE','acc':'#E67E22'}

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#FFFFFF')
gs  = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.38)

# A — ROC curves
ax1 = fig.add_subplot(gs[0, 0])
fpr_s, tpr_s, _ = roc_curve(y_arr, oof_probs)
fpr_l, tpr_l, _ = roc_curve(y_arr, loo_probs)
ax1.fill_between(fpr_s, tpr_s, alpha=0.12, color=PAL['pos'])
ax1.plot(fpr_s, tpr_s, color=PAL['pos'], lw=2.5, label=f'10-Fold SKF (AUC={auc_skf:.3f})')
ax1.plot(fpr_l, tpr_l, color=PAL['acc'], lw=2,   ls='--', label=f'LOO (AUC={loo_auc:.3f})')
ax1.plot([0,1],[0,1],'k--',lw=1,alpha=0.5)
ax1.set(xlabel='False Positive Rate', ylabel='True Positive Rate',
        title='ROC Curves\nStratified K-Fold vs LOO')
ax1.legend(fontsize=10); ax1.set_facecolor(PAL['bg']); ax1.grid(True, color=PAL['grid'])
ax1.title.set_fontsize(13); ax1.title.set_fontweight('bold')

# B — Confusion matrix
ax2 = fig.add_subplot(gs[0, 1])
ConfusionMatrixDisplay(agg_cm, display_labels=['Survived','Died']).plot(
    ax=ax2, colorbar=False, cmap='Reds')
ax2.set_title('Aggregated Confusion Matrix\n(10-Fold OOF)', fontsize=13, fontweight='bold')

# C — Feature importance (all 12)
ax3 = fig.add_subplot(gs[0, 2])
cols_f = [PAL['pos'] if i<3 else PAL['neg'] for i in range(len(feat_imp))]
ax3.barh(feat_imp.index[::-1], feat_imp.values[::-1], color=cols_f[::-1], edgecolor='white')
ax3.set(xlabel='Importance Score', title='Feature Importances\n(Gradient Boosting)')
ax3.set_facecolor(PAL['bg']); ax3.grid(True, axis='x', color=PAL['grid'])
ax3.title.set_fontsize(13); ax3.title.set_fontweight('bold')

# D — AUC per fold
ax4 = fig.add_subplot(gs[1, 0])
fa = cv_results['test_roc_auc']
fn = np.arange(1, len(fa)+1)
ax4.bar(fn, fa, color=[PAL['pos'] if a>=fa.mean() else PAL['neg'] for a in fa],
        edgecolor='white', alpha=0.85)
ax4.axhline(fa.mean(), color='red', ls='--', lw=2, label=f'Mean AUC={fa.mean():.3f}')
ax4.set(xlabel='Fold', ylabel='ROC-AUC', title='AUC per CV Fold\n(Stratified 10-Fold)')
ax4.set_ylim(0,1.1); ax4.set_xticks(fn); ax4.legend(fontsize=10)
ax4.set_facecolor(PAL['bg']); ax4.grid(True, axis='y', color=PAL['grid'])
ax4.title.set_fontsize(13); ax4.title.set_fontweight('bold')

# E — Class distribution
ax5 = fig.add_subplot(gs[1, 1])
cc = pd.Series(y_arr).value_counts().sort_index()
bars = ax5.bar(['Survived','Died'], cc.values, color=[PAL['neg'],PAL['pos']], edgecolor='white')
for b, v in zip(bars, cc.values):
    ax5.text(b.get_x()+b.get_width()/2, b.get_height()+1.5,
             f'n={v}', ha='center', fontsize=12, fontweight='bold')
ax5.set(ylabel='Count', title='Class Distribution')
ax5.set_facecolor(PAL['bg']); ax5.grid(True, axis='y', color=PAL['grid'])
ax5.title.set_fontsize(13); ax5.title.set_fontweight('bold')

# F — Summary table
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
tdata = [['Method','ROC-AUC','Accuracy'],
         ['Strat. K-Fold (10)', f"{cv_results['test_roc_auc'].mean():.4f}",
          f"{(oof_preds==y_arr).mean():.4f}"],
         ['Leave-One-Out', f'{loo_auc:.4f}', f'{loo_acc:.4f}'],
         ['','',''], ['Metric (SKF)','Mean','± Std']]
for m in ['roc_auc','f1','precision','recall']:
    s = cv_results[f'test_{m}']
    tdata.append([m.upper(), f'{s.mean():.4f}', f'±{s.std():.4f}'])
tbl = ax6.table(cellText=tdata[1:], colLabels=tdata[0], cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1.2,2.0)
for (r,c), cell in tbl.get_celld().items():
    if r==0: cell.set_facecolor(PAL['pos']); cell.set_text_props(color='white',fontweight='bold')
    elif r%2==0: cell.set_facecolor('#FDEDEC')
    cell.set_edgecolor('#DDDDDD')
ax6.set_title('Validation Summary\nSKF vs LOO', fontsize=13, fontweight='bold', pad=60)

fig.suptitle('Heart Failure Survival Classification Pipeline\n'
             'Gradient Boosting + Stratified K-Fold + Leave-One-Out CV  |  299 Patients, 12 Features',
             fontsize=15, fontweight='bold', y=0.98)
plt.savefig('/mnt/user-data/outputs/heart_failure_pipeline.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n✅ Saved: heart_failure_pipeline.png  |  SKF AUC={auc_skf:.4f}  LOO AUC={loo_auc:.4f}")
