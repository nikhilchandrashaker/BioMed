"""
╔══════════════════════════════════════════════════════════════════╗
║              BREAST CANCER RECURRENCE PREDICTION                 ║
║  Dataset : UCI Breast Cancer (Ljubljana / Oncology Institute)    ║
║  Target  : recurrence-events vs no-recurrence-events            ║
║  Model   : Random Forest Classifier (class-weight balanced)      ║
╚══════════════════════════════════════════════════════════════════╝
Usage: python breast_cancer_model.py
A file picker will open — select your breast-cancer.data file.
Outputs: breast_cancer_results.png saved to same folder as the data file.
"""

import tkinter as tk
from tkinter import filedialog
import os, pandas as pd, numpy as np, matplotlib, warnings
matplotlib.use('Agg'); warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer

# ── FILE PICKER ───────────────────────────────────────────────────────────────
root = tk.Tk(); root.withdraw(); root.lift()
DATA_PATH = filedialog.askopenfilename(
    title="Select breast-cancer.data",
    filetypes=[("Data files", "*.data"), ("All files", "*.*")])
if not DATA_PATH:
    raise SystemExit("No file selected.")
OUT_DIR = os.path.dirname(DATA_PATH)

# ── CONFIG ───────────────────────────────────────────────────────────────────
TEST_SIZE    = 0.20
N_TREES      = 300
RANDOM_STATE = 42
BCOLS = ['class','age','menopause','tumor-size','inv-nodes',
         'node-caps','deg-malig','breast','breast-quad','irradiat']

# ── PALETTE ──────────────────────────────────────────────────────────────────
ACCENT = '#00d4aa'; PINK = '#ff6b9d'; PURPLE = '#a78bfa'; ORANGE = '#ffa07a'
BG = '#161b22'; DARK = '#0d1117'; GRID = '#30363d'; TEXT = '#e6edf3'

# ── LOAD & PREPROCESS ────────────────────────────────────────────────────────
print("Loading breast cancer dataset...")
df = pd.read_csv(DATA_PATH, header=None, names=BCOLS)
df.replace('?', np.nan, inplace=True)

y = (df['class'] == 'recurrence-events').astype(int).values
FEATS = [c for c in BCOLS if c != 'class']
X_raw = pd.get_dummies(df[FEATS], dummy_na=False)
feat_names = list(X_raw.columns)
X = SimpleImputer(strategy='most_frequent').fit_transform(X_raw.values.astype(float))

print(f"  Samples: {len(y)} | Encoded Features: {len(feat_names)}")
print(f"  Class balance — No-recurrence: {(y==0).sum()} | Recurrence: {(y==1).sum()}")

# ── TRAIN / EVALUATE ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

clf = RandomForestClassifier(n_estimators=N_TREES, class_weight='balanced',
                             random_state=RANDOM_STATE)
clf.fit(X_train, y_train)

y_pred  = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]
auc     = roc_auc_score(y_test, y_proba)
cv      = cross_val_score(clf, X, y, cv=StratifiedKFold(5, shuffle=True,
                          random_state=RANDOM_STATE), scoring='roc_auc')
report  = classification_report(y_test, y_pred, zero_division=0,
                                 target_names=['No-Recurrence','Recurrence'])
cm      = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)
importances = pd.Series(clf.feature_importances_, index=feat_names).sort_values(ascending=False)

print("\n── Classification Report ──")
print(report)
print(f"ROC-AUC : {auc:.4f}")
print(f"CV AUC  : {cv.mean():.4f} ± {cv.std():.4f}")
print(f"\nTop 5 Predictive Features:")
for feat, score in importances.head(5).items():
    print(f"  {feat:<40} {score:.4f}")

# ── VISUALIZE ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10), facecolor=DARK)
fig.suptitle('Breast Cancer Recurrence Prediction — Random Forest',
             fontsize=18, fontweight='bold', color='white', y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38,
                       left=0.06, right=0.97, top=0.93, bottom=0.07)

def _style(ax, title=''):
    ax.set_facecolor(BG); ax.tick_params(colors=TEXT)
    ax.spines[:].set_color(GRID)
    for lbl in ax.get_xticklabels()+ax.get_yticklabels(): lbl.set_color(TEXT)
    if title: ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=7)
    ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)

ax0 = fig.add_subplot(gs[0, 0])
vals = [(y==0).sum(), (y==1).sum()]
wedges, texts, ats = ax0.pie(vals, labels=['No-Recurrence','Recurrence'],
    autopct='%1.0f%%', colors=[ACCENT, PINK], startangle=90,
    textprops=dict(color=TEXT, fontsize=10),
    wedgeprops=dict(edgecolor=DARK, linewidth=2))
for at in ats: at.set_color(DARK); at.set_fontweight('bold')
ax0.set_facecolor(BG); _style(ax0, 'Class Distribution')

ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(fpr, tpr, color=ACCENT, lw=2.5, label=f'AUC = {auc:.3f}')
ax1.plot([0,1],[0,1],'--', color=GRID, lw=1)
ax1.fill_between(fpr, tpr, alpha=0.15, color=ACCENT)
ax1.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=11)
_style(ax1, 'ROC Curve'); ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate')

ax2 = fig.add_subplot(gs[0, 2])
ax2.bar(range(1,6), cv, color=PURPLE, width=0.5, edgecolor='none')
ax2.axhline(cv.mean(), color=ORANGE, lw=2, linestyle='--', label=f'Mean={cv.mean():.3f}')
ax2.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=10)
ax2.set_xticks(range(1,6)); ax2.set_xticklabels([f'Fold {i}' for i in range(1,6)])
ax2.set_ylim(0,1); _style(ax2, '5-Fold CV AUC'); ax2.set_ylabel('AUC')

ax3 = fig.add_subplot(gs[1, 0])
ax3.imshow(cm, cmap='GnBu', aspect='auto')
for i in range(2):
    for j in range(2):
        ax3.text(j, i, cm[i,j], ha='center', va='center',
                 color='white', fontsize=16, fontweight='bold')
ax3.set_xticks([0,1]); ax3.set_yticks([0,1])
ax3.set_xticklabels(['Pred No-Rec','Pred Rec']); ax3.set_yticklabels(['Act No-Rec','Act Rec'])
_style(ax3, 'Confusion Matrix')

ax4 = fig.add_subplot(gs[1, 1:])
top = importances.head(12)
clrs = [ACCENT if i<3 else PINK if i<6 else PURPLE for i in range(len(top))]
ax4.barh(range(len(top)), top.values, color=clrs, edgecolor='none')
ax4.set_yticks(range(len(top)))
ax4.set_yticklabels(top.index, fontsize=8); ax4.invert_yaxis()
for i, v in enumerate(top.values):
    ax4.text(v+0.001, i, f'{v:.3f}', va='center', color=TEXT, fontsize=8)
_style(ax4, 'Top 12 Feature Importances'); ax4.set_xlabel('Importance Score')

out = os.path.join(OUT_DIR, 'breast_cancer_results.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()
print(f"\nVisualization saved → {out}")
