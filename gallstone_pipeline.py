"""
=============================================================
  GALLSTONE RISK CLASSIFICATION PIPELINE
  Dataset: UCI Gallstone (319 patients, 38 features)
  Target: Gallstone Status (0=no gallstone, 1=gallstone)
  Techniques: Stratified K-Fold CV, SMOTE, Random Forest
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
df = pd.read_csv('/tmp/datasets/gallstone/gallstone.csv')
df = df.apply(pd.to_numeric, errors='coerce')

print(f"Dataset shape: {df.shape}")
TARGET = 'Gallstone Status'
print(f"Target distribution:\n{df[TARGET].value_counts()}")

y_arr = df[TARGET].values.astype(int)
X     = df.drop(columns=[TARGET])

# Short feature names for plots
short_names = {
    'Body Mass Index (BMI)':             'BMI',
    'Total Body Water (TBW)':            'TBW',
    'Extracellular Water (ECW)':         'ECW',
    'Intracellular Water (ICW)':         'ICW',
    'Extracellular Fluid/Total Body Water (ECF/TBW)': 'ECF/TBW',
    'Total Body Fat Ratio (TBFR) (%)':   'TBFR%',
    'Lean Mass (LM) (%)':                'Lean Mass%',
    'Body Protein Content (Protein) (%)':'Protein%',
    'Visceral Fat Rating (VFR)':         'VFR',
    'Bone Mass (BM)':                    'Bone Mass',
    'Muscle Mass (MM)':                  'Muscle Mass',
    'Total Fat Content (TFC)':           'TFC',
    'Visceral Fat Area (VFA)':           'VFA',
    'Hepatic Fat Accumulation (HFA)':    'Hep. Fat',
    'Visceral Muscle Area (VMA) (Kg)':   'VMA',
    'Total Cholesterol (TC)':            'Cholesterol',
    'Low Density Lipoprotein (LDL)':     'LDL',
    'High Density Lipoprotein (HDL)':    'HDL',
    'Aspartat Aminotransferaz (AST)':    'AST',
    'Alanin Aminotransferaz (ALT)':      'ALT',
    'Alkaline Phosphatase (ALP)':        'ALP',
    'Glomerular Filtration Rate (GFR)':  'GFR',
    'C-Reactive Protein (CRP)':          'CRP',
    'Coronary Artery Disease (CAD)':     'CAD',
    'Diabetes Mellitus (DM)':            'Diabetes',
    'Obesity (%)':                       'Obesity%',
}
X = X.rename(columns=short_names)
X_arr = X.values.astype(float)

# ─────────────────────────────────────────────
# 2. PIPELINE
# ─────────────────────────────────────────────
clf = ImbPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('smote',   SMOTE(random_state=42, k_neighbors=5)),
    ('model',   RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight='balanced',
        random_state=42, n_jobs=-1))
])

# ─────────────────────────────────────────────
# 3. STRATIFIED 10-FOLD CV
# ─────────────────────────────────────────────
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_results = cross_validate(clf, X_arr, y_arr, cv=skf,
    scoring=['roc_auc','f1','precision','recall'],
    return_train_score=True)

print("\n" + "="*50)
print("STRATIFIED 10-FOLD CV RESULTS")
print("="*50)
for m in ['roc_auc','f1','precision','recall']:
    s = cv_results[f'test_{m}']
    print(f"{m.upper():12s}: {s.mean():.4f} ± {s.std():.4f}")

# ─────────────────────────────────────────────
# 4. OOF PREDICTIONS
# ─────────────────────────────────────────────
oof_probs = np.zeros(len(y_arr))
oof_preds = np.zeros(len(y_arr))
conf_mats = []
for tr, va in skf.split(X_arr, y_arr):
    clf.fit(X_arr[tr], y_arr[tr])
    oof_probs[va] = clf.predict_proba(X_arr[va])[:, 1]
    oof_preds[va] = clf.predict(X_arr[va])
    conf_mats.append(confusion_matrix(y_arr[va], clf.predict(X_arr[va])))

agg_cm  = sum(conf_mats)
auc_skf = roc_auc_score(y_arr, oof_probs)

# Feature importance on full data
imp = SimpleImputer(strategy='median')
sc  = StandardScaler()
X_imp = imp.fit_transform(X_arr)
X_sc  = sc.fit_transform(X_imp)
sm = SMOTE(random_state=42, k_neighbors=5)
X_res, y_res = sm.fit_resample(X_sc, y_arr)
rf_full = RandomForestClassifier(n_estimators=300, max_depth=8,
                                  class_weight='balanced', random_state=42)
rf_full.fit(X_res, y_res)
feat_imp = pd.Series(rf_full.feature_importances_,
                     index=X.columns).sort_values(ascending=False)

# ─────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────
PAL = {'pos':'#7D3C98','neg':'#2471A3','bg':'#F8F9FA','grid':'#ECEDEE','acc':'#1ABC9C'}

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#FFFFFF')
gs  = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.38)

# A — ROC curve
ax1 = fig.add_subplot(gs[0, 0])
fpr, tpr, _ = roc_curve(y_arr, oof_probs)
ax1.fill_between(fpr, tpr, alpha=0.15, color=PAL['pos'])
ax1.plot(fpr, tpr, color=PAL['pos'], lw=2.5, label=f'ROC (AUC={auc_skf:.3f})')
ax1.plot([0,1],[0,1],'k--',lw=1,alpha=0.5)
ax1.set(xlabel='False Positive Rate', ylabel='True Positive Rate',
        title='ROC Curve (Out-of-Fold)\n10-Fold Stratified CV')
ax1.legend(fontsize=10); ax1.set_facecolor(PAL['bg']); ax1.grid(True, color=PAL['grid'])
ax1.title.set_fontsize(13); ax1.title.set_fontweight('bold')

# B — Confusion matrix
ax2 = fig.add_subplot(gs[0, 1])
ConfusionMatrixDisplay(agg_cm, display_labels=['No Gallstone','Gallstone']).plot(
    ax=ax2, colorbar=False, cmap='Purples')
ax2.set_title('Aggregated Confusion Matrix\n(10-Fold OOF)', fontsize=13, fontweight='bold')

# C — Top 15 feature importance
ax3 = fig.add_subplot(gs[0, 2])
top15 = feat_imp.head(15)
cols_f = [PAL['pos'] if i<5 else PAL['neg'] for i in range(15)]
ax3.barh(top15.index[::-1], top15.values[::-1], color=cols_f[::-1], edgecolor='white')
ax3.set(xlabel='Importance Score', title='Top 15 Feature Importances\n(Random Forest + SMOTE)')
ax3.set_facecolor(PAL['bg']); ax3.grid(True, axis='x', color=PAL['grid'])
ax3.title.set_fontsize(13); ax3.title.set_fontweight('bold')

# D — CV score distribution boxplot
ax4 = fig.add_subplot(gs[1, 0])
mp = {'ROC-AUC':cv_results['test_roc_auc'], 'F1':cv_results['test_f1'],
      'Precision':cv_results['test_precision'], 'Recall':cv_results['test_recall']}
bp = ax4.boxplot(mp.values(), labels=mp.keys(), patch_artist=True,
                 medianprops={'color':'white','linewidth':2})
for patch, color in zip(bp['boxes'], [PAL['pos'],PAL['neg'],PAL['acc'],'#E74C3C']):
    patch.set_facecolor(color); patch.set_alpha(0.75)
ax4.set(ylabel='Score', title='CV Metric Distribution\n(10-Fold)')
ax4.set_ylim(0,1.1); ax4.set_facecolor(PAL['bg']); ax4.grid(True, axis='y', color=PAL['grid'])
ax4.title.set_fontsize(13); ax4.title.set_fontweight('bold')

# E — Class distribution
ax5 = fig.add_subplot(gs[1, 1])
cc = pd.Series(y_arr).value_counts().sort_index()
bars = ax5.bar(['No Gallstone (0)','Gallstone (1)'], cc.values,
               color=[PAL['neg'],PAL['pos']], edgecolor='white')
for b, v in zip(bars, cc.values):
    ax5.text(b.get_x()+b.get_width()/2, b.get_height()+1.5,
             f'n={v}', ha='center', fontsize=11, fontweight='bold')
ax5.set(ylabel='Count', title='Class Distribution\n(SMOTE Applied for Training)')
ax5.set_facecolor(PAL['bg']); ax5.grid(True, axis='y', color=PAL['grid'])
ax5.title.set_fontsize(13); ax5.title.set_fontweight('bold')

# F — Score table
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
tdata = []
for m in ['roc_auc','f1','precision','recall']:
    s = cv_results[f'test_{m}']
    tdata.append([m.upper(), f'{s.mean():.4f}', f'{s.std():.4f}',
                  f'{s.min():.4f}', f'{s.max():.4f}'])
tbl = ax6.table(cellText=tdata,
                colLabels=['Metric','Mean','Std','Min','Max'],
                cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2,2.0)
for (r,c), cell in tbl.get_celld().items():
    if r==0: cell.set_facecolor(PAL['pos']); cell.set_text_props(color='white',fontweight='bold')
    elif r%2==0: cell.set_facecolor('#F5EEF8')
    cell.set_edgecolor('#DDDDDD')
ax6.set_title('Cross-Validation Summary', fontsize=13, fontweight='bold', pad=60)

fig.suptitle('Gallstone Risk Classification Pipeline\n'
             'Random Forest + SMOTE + Stratified 10-Fold CV  |  319 Patients, 38 Features',
             fontsize=15, fontweight='bold', y=0.98)
plt.savefig('/mnt/user-data/outputs/gallstone_pipeline.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n✅ Saved: gallstone_pipeline.png  |  OOF AUC={auc_skf:.4f}")
