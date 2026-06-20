"""
03_train_and_export.py  ─  Train & Export  (v6 – 27 features, 10 classes)
==========================================================================
Labels: 0=Normal,  1=Imb_Mild,  2=Imb_Mod,  3=Imb_Sev,
        5=Blade_Damage,
        7=Loose_Mild, 8=Loose_Mod, 9=Loose_Sev,
        10=Bearing_Contamination,
        11=Air_Obstruction

v6 changes (everything else identical to v5):
  - LABEL_NAMES extended with labels 10 and 11
  - Blade-damage confusion check unchanged
  - Bearing contamination confusion check added
    (checks it is not confused with Looseness classes)
  - Air obstruction confusion check added
    (checks it is not confused with Normal or Blade_Damage)
"""

import os, pickle
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble        import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import classification_report, confusion_matrix, accuracy_score

SAVE_DIR = r'C:\Users\Dell\OneDrive\Documents\Fan_Project'

LABEL_NAMES = {
    0:'Normal', 1:'Imbalance_Mild', 2:'Imbalance_Moderate', 3:'Imbalance_Severe',
    5:'Blade_Damage',
    7:'Looseness_Mild', 8:'Looseness_Moderate', 9:'Looseness_Severe',
    11:'Bearing_Contamination',
    13:'Air_Obstruction',
}

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading features_dataset.csv …")
df = pd.read_csv(os.path.join(SAVE_DIR, 'features_dataset.csv'))
print(f"  {len(df)} samples | {df.shape[1]-1} features")

print("\nClass distribution:")
for lbl, cnt in df['label'].value_counts().sort_index().items():
    print(f"  {LABEL_NAMES.get(lbl,str(lbl)):26s} (label={lbl}): {cnt:5d}")

feat_cols = [c for c in df.columns if c != 'label']
X = df[feat_cols].values
y_orig = df['label'].values.astype(int)

# Remap sparse → contiguous for micromlgen
present   = sorted(np.unique(y_orig))
remap     = {orig: new for new, orig in enumerate(present)}
unremap   = {v: k for k, v in remap.items()}
y         = np.array([remap[l] for l in y_orig])
class_names = [LABEL_NAMES.get(unremap[i], str(i)) for i in range(len(present))]

print(f"\nESP32 index → class name:")
for i, name in enumerate(class_names):
    print(f"  {i} → {name}  (original label={unremap[i]})")

# ── Split & scale ──────────────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_tr   = scaler.fit_transform(X_tr)
X_te   = scaler.transform(X_te)
print(f"\nTrain: {len(X_tr)}  |  Test: {len(X_te)}")

# ── Compare classifiers ────────────────────────────────────────────────────────
candidates = {
    'RandomForest': RandomForestClassifier(
        n_estimators=80, max_depth=14, min_samples_leaf=2,
        class_weight='balanced', random_state=42, n_jobs=-1),
    'ExtraTrees': ExtraTreesClassifier(
        n_estimators=80, max_depth=14, min_samples_leaf=2,
        class_weight='balanced', random_state=42, n_jobs=-1),
}

print("\n5-fold CV (F1-macro):")
best_name, best_score, best_clf = None, 0.0, None
for name, clf in candidates.items():
    cv = cross_val_score(clf, X_tr, y_tr, cv=5, scoring='f1_macro', n_jobs=-1)
    print(f"  {name:15s}: {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")
    if cv.mean() > best_score:
        best_score, best_name, best_clf = cv.mean(), name, clf

print(f"\n→ Using {best_name}")
best_clf.fit(X_tr, y_tr)
y_pred = best_clf.predict(X_te)
acc    = accuracy_score(y_te, y_pred)
print(f"Test Accuracy: {acc*100:.2f}%")
print(classification_report(y_te, y_pred, target_names=class_names))

# Confusion matrix
cm  = confusion_matrix(y_te, y_pred)
fig, ax = plt.subplots(figsize=(13, 9))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_title(f'Confusion Matrix — {best_name}  ({acc*100:.1f}%)')
ax.set_ylabel('Actual');  ax.set_xlabel('Predicted')
plt.xticks(rotation=45, ha='right');  plt.tight_layout()
cm_path = os.path.join(SAVE_DIR, 'confusion_matrix.png')
fig.savefig(cm_path, dpi=150);  plt.show()

# Feature importance
BLADE_FEATS = {'az_ax_rms_ratio','abs_autocorr_az_1rev','abs_autocorr_az_2rev'}
importances = best_clf.feature_importances_
order = np.argsort(importances)[::-1]
fig2, ax2 = plt.subplots(figsize=(14, 5))
colors = ['#d62728' if feat_cols[i] in BLADE_FEATS else '#1f77b4' for i in order]
ax2.bar(range(len(feat_cols)), importances[order], color=colors)
ax2.set_xticks(range(len(feat_cols)))
ax2.set_xticklabels([feat_cols[i] for i in order], rotation=55, ha='right', fontsize=8)
ax2.set_title('Feature Importance  (red = bent-blade features)')
plt.tight_layout()
fig2.savefig(os.path.join(SAVE_DIR, 'feature_importance.png'), dpi=150);  plt.show()

print("\nTop-10 features:")
for rank, i in enumerate(order[:10], 1):
    star = ' ★' if feat_cols[i] in BLADE_FEATS else ''
    print(f"  {rank:2d}. {feat_cols[i]:28s} {importances[i]:.4f}{star}")

# ── Blade damage confusion check (unchanged from v5) ───────────────────────────
bd_idx  = remap.get(5, None)
imb_idx = [remap[l] for l in [1, 2, 3] if l in remap]
if bd_idx is not None:
    bd_mask  = y_te == bd_idx
    imb_mask = np.isin(y_te, imb_idx)
    bd_as_imb  = np.isin(y_pred[bd_mask],  imb_idx).sum()
    imb_as_bd  = (y_pred[imb_mask] == bd_idx).sum()
    n_bd, n_imb = bd_mask.sum(), imb_mask.sum()
    print(f"\n=== BENT BLADE vs IMBALANCE ===")
    print(f"  BD  → predicted Imbalance: {bd_as_imb}/{n_bd} "
          f"({bd_as_imb/max(n_bd,1)*100:.1f}%)  "
          f"{'✓' if bd_as_imb/max(n_bd,1)<0.05 else '✗'}")
    print(f"  Imb → predicted BD:        {imb_as_bd}/{n_imb} "
          f"({imb_as_bd/max(n_imb,1)*100:.1f}%)  "
          f"{'✓' if imb_as_bd/max(n_imb,1)<0.05 else '✗'}")

# ── Bearing contamination confusion check (new) ────────────────────────────────
brg_idx   = remap.get(10, None)
loose_idx = [remap[l] for l in [7, 8, 9] if l in remap]
norm_idx  = remap.get(0, None)
if brg_idx is not None:
    brg_mask   = y_te == brg_idx
    loose_mask = np.isin(y_te, loose_idx)
    brg_as_loose  = np.isin(y_pred[brg_mask],  loose_idx).sum()
    loose_as_brg  = (y_pred[loose_mask] == brg_idx).sum()
    brg_as_norm   = (y_pred[brg_mask] == norm_idx).sum() if norm_idx is not None else 0
    n_brg, n_loose = brg_mask.sum(), loose_mask.sum()
    print(f"\n=== BEARING CONTAMINATION vs LOOSENESS ===")
    print(f"  BRG → predicted Looseness: {brg_as_loose}/{n_brg} "
          f"({brg_as_loose/max(n_brg,1)*100:.1f}%)  "
          f"{'✓' if brg_as_loose/max(n_brg,1)<0.05 else '✗'}")
    print(f"  BRG → predicted Normal:    {brg_as_norm}/{n_brg} "
          f"({brg_as_norm/max(n_brg,1)*100:.1f}%)  "
          f"{'✓' if brg_as_norm/max(n_brg,1)<0.05 else '✗'}")
    print(f"  Loose → predicted BRG:     {loose_as_brg}/{n_loose} "
          f"({loose_as_brg/max(n_loose,1)*100:.1f}%)  "
          f"{'✓' if loose_as_brg/max(n_loose,1)<0.05 else '✗'}")

# ── Air obstruction confusion check (new) ─────────────────────────────────────
air_idx = remap.get(11, None)
if air_idx is not None:
    air_mask  = y_te == air_idx
    # Risk: confused with Normal (both have high az_ax_rms_ratio)
    # or with Blade_Damage (both have high autocorr_az)
    air_as_norm = (y_pred[air_mask] == norm_idx).sum() if norm_idx is not None else 0
    air_as_bd   = (y_pred[air_mask] == bd_idx).sum()   if bd_idx  is not None else 0
    n_air = air_mask.sum()
    if norm_idx is not None:
        norm_mask    = y_te == norm_idx
        norm_as_air  = (y_pred[norm_mask] == air_idx).sum()
        n_norm       = norm_mask.sum()
    else:
        norm_as_air = n_norm = 0
    print(f"\n=== AIR OBSTRUCTION CONFUSION CHECK ===")
    print(f"  AIR → predicted Normal:     {air_as_norm}/{n_air} "
          f"({air_as_norm/max(n_air,1)*100:.1f}%)  "
          f"{'✓' if air_as_norm/max(n_air,1)<0.05 else '✗'}")
    print(f"  AIR → predicted Blade_Dmg: {air_as_bd}/{n_air} "
          f"({air_as_bd/max(n_air,1)*100:.1f}%)  "
          f"{'✓' if air_as_bd/max(n_air,1)<0.05 else '✗'}")
    print(f"  Normal → predicted AIR:    {norm_as_air}/{n_norm} "
          f"({norm_as_air/max(n_norm,1)*100:.1f}%)  "
          f"{'✓' if norm_as_air/max(n_norm,1)<0.05 else '✗'}")

# ── Save ───────────────────────────────────────────────────────────────────────
with open(os.path.join(SAVE_DIR, 'rf_model.pkl'), 'wb') as f: pickle.dump(best_clf, f)
with open(os.path.join(SAVE_DIR, 'scaler.pkl'),   'wb') as f: pickle.dump(scaler, f)
print("\nModel and scaler saved.")

# ── ESP32 export ───────────────────────────────────────────────────────────────
print("\nExporting to ESP32 …")
try:
    from micromlgen import port
    c_code = port(best_clf, classname='FaultClassifier')
    hpath  = os.path.join(SAVE_DIR, 'FaultClassifier.h')
    open(hpath,'w').write(c_code)
    print(f"  Saved: {hpath}  ({os.path.getsize(hpath)/1024:.1f} KB)")
except Exception as e:
    print(f"  micromlgen failed: {e}")

# ── Print arrays for Arduino sketch ───────────────────────────────────────────
print(f"\n// ── PASTE INTO ARDUINO SKETCH ──────────────────────────")
print(f"const int   N_FEATURES  = {len(feat_cols)};")
print(f"const int   N_CLASSES   = {len(class_names)};")
print(f'const char* CLASS_NAMES[] = {{', end='')
print(', '.join(f'"{n}"' for n in class_names), end='')
print("};")
print("float scaler_mean[] = {", end='')
print(', '.join(f'{v:.8f}f' for v in scaler.mean_), end='')
print("};")
print("float scaler_std[] = {", end='')
print(', '.join(f'{v:.8f}f' for v in scaler.scale_), end='')
print("};")
print("\n// Feature order:")
for i, name in enumerate(feat_cols):
    star = '  // ★' if name in BLADE_FEATS else ''
    print(f"// [{i:2d}] {name}{star}")
