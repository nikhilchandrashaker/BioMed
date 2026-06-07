"""
=============================================================
  DIABETES HYPOGLYCEMIA CLASSIFICATION PIPELINE
  Dataset: UCI Diabetes Log Data (70 outpatient records)
  Target: Hypoglycemic episode (code 65 presence per patient)
  Techniques: Feature engineering from time-series event codes,
              Stratified K-Fold CV, Leave-One-Out CV,
              Gradient Boosting
=============================================================
"""

import numpy as np
import pandas as pd
import os
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

# ─────────────────────────────────────────────────────────
# EVENT-CODE REFERENCE (from Data-Codes file)
# 33 = Regular insulin dose          34 = NPH insulin dose
# 35 = UltraLente insulin dose
# 48,57 = Unspecified blood-glucose  58 = Pre-breakfast BG
# 59 = Post-breakfast BG             60 = Pre-lunch BG
# 61 = Post-lunch BG                 62 = Pre-supper BG
# 63 = Post-supper BG                64 = Pre-snack BG
# 65 = Hypoglycemic symptoms  ← TARGET
# 66 = Typical meal    67 = More meal    68 = Less meal
# 69 = Typical exercise  70 = More exercise  71 = Less exercise
# ─────────────────────────────────────────────────────────

DATA_DIR  = '/tmp/datasets/diabetes/Diabetes-Data/'
BG_CODES  = {48, 57, 58, 59, 60, 61, 62, 63, 64}
INS_CODES = {33, 34, 35}
HYPO_CODE = 65

# ─────────────────────────────────────────────
# 1. BUILD PATIENT-LEVEL FEATURE TABLE
# ─────────────────────────────────────────────
records = []
for fname in sorted(f for f in os.listdir(DATA_DIR) if f.startswith('data-')):
    rows = []
    try:
        with open(os.path.join(DATA_DIR, fname), 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    try:
                        code  = int(parts[2])
                        value = float(parts[3])
                        rows.append((code, value))
                    except ValueError:
                        pass
    except Exception:
        continue

    if len(rows) < 10:
        continue

    df_p = pd.DataFrame(rows, columns=['code','val'])
    bg   = df_p[df_p['code'].isin(BG_CODES)]['val']
    ins  = df_p[df_p['code'].isin(INS_CODES)]['val']
    hypo = (df_p['code'] == HYPO_CODE).sum()

    records.append({
        'patient':            fname,
        'n_records':          len(rows),
        # Blood-glucose stats
        'bg_mean':            bg.mean() if len(bg) else np.nan,
        'bg_std':             bg.std()  if len(bg)>1 else np.nan,
        'bg_min':             bg.min()  if len(bg) else np.nan,
        'bg_max':             bg.max()  if len(bg) else np.nan,
        'bg_low_count':       (bg < 70 ).sum(),
        'bg_high_count':      (bg > 200).sum(),
        'bg_range':           (bg.max()-bg.min()) if len(bg)>1 else np.nan,
        # Insulin stats
        'insulin_total':      ins.sum()  if len(ins) else 0,
        'insulin_mean':       ins.mean() if len(ins) else np.nan,
        'regular_doses':      (df_p['code']==33).sum(),
        'nph_doses':          (df_p['code']==34).sum(),
        'ultralente_doses':   (df_p['code']==35).sum(),
        # Meal / exercise behaviour
        'n_meals':            df_p['code'].isin({66,67,68}).sum(),
        'excess_meals':       (df_p['code']==67).sum(),
        'skipped_meals':      (df_p['code']==68).sum(),
        'n_exercise':         df_p['code'].isin({69,70,71}).sum(),
        'excess_exercise':    (df_p['code']==70).sum(),
        # Target
        'hypo_events':        hypo,
        'hypo_label':         int(hypo > 0),
    })

df_feat = pd.DataFrame(records)
print(f"Patients loaded: {len(df_feat)}")
print(f"\nHypoglycemic label distribution:\n{df_feat['hypo_label'].value_counts()}")

FEAT_COLS = [c for c in df_feat.columns if c not in ('patient','hypo_events','hypo_label')]
y_arr = df_feat['hypo_label'].values
X     = df_feat[FEAT_COLS]
X_arr = X.values.astype(float)

# ─────────────────────────────────────────────
# 2. PIPELINE
# ─────────────────────────────────────────────
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.05,
        max_depth=2, subsample=0.8, random_state=42))
])

# ─────────────────────────────────────────────
# 3. STRATIFIED K-FOLD (k=5, capped by minority)
# ─────────────────────────────────────────────
n_splits = min(5, int(y_arr.sum()), int((1-y_arr).sum()))
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
cv_results = cross_validate(pipe, X_arr, y_arr, cv=skf,
    scoring=['roc_auc','f1','precision','recall'],
    return_train_score=True)

print(f"\n{'='*50}")
print(f"STRATIFIED {n_splits}-FOLD CV RESULTS")
print(f"{'='*50}")
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
                     index=FEAT_COLS).sort_values(ascending=False)

# BG distribution by class (for panel E)
bg_hypo    = df_feat[df_feat['hypo_label']==1]['bg_mean'].dropna()
bg_no_hypo = df_feat[df_feat['hypo_label']==0]['bg_mean'].dropna()

# ─────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────
PAL = {'pos':'#E67E22','neg':'#2980B9','bg':'#F8F9FA','grid':'#ECEDEE','acc':'#27AE60'}

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#FFFFFF')
gs  = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.38)

# A — ROC curves (SKF + LOO)
ax1 = fig.add_subplot(gs[0, 0])
fpr_s, tpr_s, _ = roc_curve(y_arr, oof_probs)
fpr_l, tpr_l, _ = roc_curve(y_arr, loo_probs)
ax1.fill_between(fpr_s, tpr_s, alpha=0.12, color=PAL['pos'])
ax1.plot(fpr_s, tpr_s, color=PAL['pos'], lw=2.5,
         label=f'{n_splits}-Fold SKF (AUC={auc_skf:.3f})')
ax1.plot(fpr_l, tpr_l, color=PAL['acc'], lw=2,   ls='--',
         label=f'LOO (AUC={loo_auc:.3f})')
ax1.plot([0,1],[0,1],'k--',lw=1,alpha=0.5)
ax1.set(xlabel='False Positive Rate', ylabel='True Positive Rate',
        title='ROC Curves\nStratified K-Fold vs LOO')
ax1.legend(fontsize=10); ax1.set_facecolor(PAL['bg']); ax1.grid(True, color=PAL['grid'])
ax1.title.set_fontsize(13); ax1.title.set_fontweight('bold')

# B — Confusion matrix
ax2 = fig.add_subplot(gs[0, 1])
ConfusionMatrixDisplay(agg_cm, display_labels=['No Hypo','Hypoglycemic']).plot(
    ax=ax2, colorbar=False, cmap='Oranges')
ax2.set_title('Aggregated Confusion Matrix\n(K-Fold OOF)', fontsize=13, fontweight='bold')

# C — Feature importance (all engineered features)
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
ax4.set(xlabel='Fold', ylabel='ROC-AUC',
        title=f'AUC per CV Fold\n(Stratified {n_splits}-Fold)')
ax4.set_ylim(0,1.1); ax4.set_xticks(fn); ax4.legend(fontsize=10)
ax4.set_facecolor(PAL['bg']); ax4.grid(True, axis='y', color=PAL['grid'])
ax4.title.set_fontsize(13); ax4.title.set_fontweight('bold')

# E — Mean BG distribution by class
ax5 = fig.add_subplot(gs[1, 1])
bins = np.linspace(min(bg_no_hypo.min(), bg_hypo.min()),
                   max(bg_no_hypo.max(), bg_hypo.max()), 16)
ax5.hist(bg_no_hypo, bins=bins, alpha=0.65, color=PAL['neg'],
         label='No Hypoglycemia', edgecolor='white')
ax5.hist(bg_hypo,    bins=bins, alpha=0.65, color=PAL['pos'],
         label='Hypoglycemia',    edgecolor='white')
ax5.axvline(bg_no_hypo.mean(), color=PAL['neg'], lw=2, ls='--')
ax5.axvline(bg_hypo.mean(),    color=PAL['pos'], lw=2, ls='--')
ax5.set(xlabel='Mean Blood Glucose (mg/dL)', ylabel='Patient Count',
        title='Mean BG Distribution\nby Hypoglycemia Status')
ax5.legend(fontsize=10); ax5.set_facecolor(PAL['bg']); ax5.grid(True, axis='y', color=PAL['grid'])
ax5.title.set_fontsize(13); ax5.title.set_fontweight('bold')

# F — Summary table
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
tdata = [['Method','ROC-AUC','Accuracy'],
         [f'Strat. K-Fold ({n_splits})', f"{cv_results['test_roc_auc'].mean():.4f}",
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
    elif r%2==0: cell.set_facecolor('#FEF9E7')
    cell.set_edgecolor('#DDDDDD')
ax6.set_title('Validation Summary\nSKF vs LOO', fontsize=13, fontweight='bold', pad=60)

fig.suptitle(
    f'Diabetes Hypoglycemia Classification Pipeline\n'
    f'Gradient Boosting + Stratified {n_splits}-Fold + LOO CV  |  '
    f'{len(df_feat)} Patients, {len(FEAT_COLS)} Engineered Features',
    fontsize=15, fontweight='bold', y=0.98)
plt.savefig('/mnt/user-data/outputs/diabetes_pipeline.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n✅ Saved: diabetes_pipeline.png  |  SKF AUC={auc_skf:.4f}  LOO AUC={loo_auc:.4f}")
