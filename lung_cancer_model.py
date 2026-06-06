"""
╔══════════════════════════════════════════════════════════════════╗
║              LUNG CANCER TYPE PREDICTION MODEL                   ║
║  Dataset : UCI Lung Cancer (Hong & Yang 1991)                    ║
║  Target  : Cancer type 1 vs types 2 & 3 (binary)               ║
║  Model   : Random Forest + Leave-One-Out CV (small dataset)      ║
╚══════════════════════════════════════════════════════════════════╝
Usage: python lung_cancer_model.py
Outputs: lung_cancer_results.png, prints metrics to console.
Place lung-cancer.data in the same directory.
Note: Only 32 samples — LOO cross-validation is used.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
import warnings; warnings.filterwarnings('ignore')

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATA_PATH    = 'lung-cancer.data'
N_TREES      = 300
RANDOM_STATE = 42

# ── PALETTE ──────────────────────────────────────────────────────────────────
ACCENT = '#58a6ff'; TEAL = '#00d4aa'; PURPLE = '#a78bfa'; ORANGE = '#ffa07a'
BG = '#161b22'; DARK = '#0d1117'; GRID = '#30363d'; TEXT = '#e6edf3'

# ── LOAD & PREPROCESS ────────────────────────────────────────────────────────
print("Loading lung cancer dataset...")
df = pd.read_csv(DATA_PATH, header=None)
df.replace('?', np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')

# Binary: class 1 vs classes 2,3
y = (df[0].values == 1).astype(int)
X = SimpleImputer(strategy='median').fit_transform(df.drop(0, axis=1).values)
feat_names = [f'Attr_{i+1}' for i in range(X.shape[1])]

print(f"  Samples: {len(y)} | Features: {X.shape[1]}")
print(f"  Class balance — Type-1: {(y==1).sum()} | Type-2/3: {(y==0).sum()}")
print("  Using Leave-One-Out CV due to small dataset size.")

# ── TRAIN / EVALUATE (LOO) ───────────────────────────────────────────────────
clf = RandomForestClassifier(n_estimators=N_TREES, class_weight='balanced',
                             random_state=RANDOM_STATE)
loo = LeaveOneOut()
preds, probs, trues = [], [], []
for ti, vi in loo.split(X):
    if len(np.unique(y[ti])) < 2:
        probs.append(0.5); preds.append(0); trues.append(y[vi][0]); continue
    clf.fit(X[ti], y[ti])
    preds.append(clf.predict(X[vi])[0])
    probs.append(clf.predict_proba(X[vi])[0, 1])
    trues.append(y[vi][0])

y_pred = np.array(preds); y_proba = np.array(probs); y_true = np.array(trues)
auc    = roc_auc_score(y_true, y_proba)
cv     = cross_val_score(clf, X, y, cv=min(5, int(y.sum())), scoring='roc_auc')
report = classification_report(y_true, y_pred, zero_division=0,
                                target_names=['Type-2/3', 'Type-1'])
cm     = confusion_matrix(y_true, y_pred)
fpr, tpr, _ = roc_curve(y_true, y_proba)

# Full-data fit for feature importances
clf.fit(X, y)
importances = pd.Series(clf.feature_importances_, index=feat_names).sort_values(ascending=False)

print("\n── Classification Report (LOO) ──")
print(report)
print(f"ROC-AUC (LOO): {auc:.4f}")
print(f"CV AUC  : {cv.mean():.4f} ± {cv.std():.4f}")
print(f"\nTop 5 Predictive Attributes:")
for feat, score in importances.head(5).items():
    print(f"  {feat:<15} {score:.4f}")

# ── VISUALIZE ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10), facecolor=DARK)
fig.suptitle('Lung Cancer Type Prediction — Random Forest (LOO-CV)',
             fontsize=18, fontweight='bold', color='white', y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38,
                       left=0.06, right=0.97, top=0.93, bottom=0.07)

def _style(ax, title=''):
    ax.set_facecolor(BG); ax.tick_params(colors=TEXT)
    ax.spines[:].set_color(GRID)
    for lbl in ax.get_xticklabels()+ax.get_yticklabels(): lbl.set_color(TEXT)
    if title: ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=7)
    ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)

# Class distribution
ax0 = fig.add_subplot(gs[0, 0])
vals = [(y==0).sum(), (y==1).sum()]
wedges, texts, ats = ax0.pie(vals, labels=['Type 2/3','Type 1'],
    autopct='%1.0f%%', colors=[TEAL, ACCENT], startangle=90,
    textprops=dict(color=TEXT, fontsize=10),
    wedgeprops=dict(edgecolor=DARK, linewidth=2))
for at in ats: at.set_color(DARK); at.set_fontweight('bold')
ax0.set_facecolor(BG); _style(ax0, 'Class Distribution')

# ROC
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(fpr, tpr, color=ACCENT, lw=2.5, label=f'AUC = {auc:.3f}')
ax1.plot([0,1],[0,1],'--', color=GRID, lw=1)
ax1.fill_between(fpr, tpr, alpha=0.15, color=ACCENT)
ax1.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=11)
_style(ax1, 'ROC Curve (LOO)'); ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate')

# CV scores
ax2 = fig.add_subplot(gs[0, 2])
ax2.bar(range(1,len(cv)+1), cv, color=PURPLE, width=0.5, edgecolor='none')
ax2.axhline(cv.mean(), color=ORANGE, lw=2, linestyle='--', label=f'Mean={cv.mean():.3f}')
ax2.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=10)
ax2.set_xticks(range(1,len(cv)+1)); ax2.set_xticklabels([f'Fold {i}' for i in range(1,len(cv)+1)])
ax2.set_ylim(0,1); _style(ax2, 'CV AUC Scores'); ax2.set_ylabel('AUC')

# Confusion matrix
ax3 = fig.add_subplot(gs[1, 0])
ax3.imshow(cm, cmap='Blues', aspect='auto')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax3.text(j, i, cm[i,j], ha='center', va='center',
                 color='white', fontsize=16, fontweight='bold')
ax3.set_xticks([0,1]); ax3.set_yticks([0,1])
ax3.set_xticklabels(['Pred 2/3','Pred 1']); ax3.set_yticklabels(['Act 2/3','Act 1'])
_style(ax3, 'Confusion Matrix (LOO)')

# Feature importances
ax4 = fig.add_subplot(gs[1, 1:])
top = importances.head(12)
clrs = [ACCENT if i<3 else TEAL if i<6 else PURPLE for i in range(len(top))]
ax4.barh(range(len(top)), top.values, color=clrs, edgecolor='none')
ax4.set_yticks(range(len(top)))
ax4.set_yticklabels(top.index, fontsize=9); ax4.invert_yaxis()
for i, v in enumerate(top.values):
    ax4.text(v+0.001, i, f'{v:.3f}', va='center', color=TEXT, fontsize=9)
_style(ax4, 'Top 12 Attribute Importances'); ax4.set_xlabel('Importance Score')

out = 'lung_cancer_results.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()
print(f"\nVisualization saved → {out}")
