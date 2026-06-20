"""
02_extract_features.py  ─  Fan Fault Feature Extraction  (v6 – 27 features, 10 classes)
=========================================================================================
Classes
-------
  0   Normal
  1   Imbalance_Mild
  2   Imbalance_Moderate
  3   Imbalance_Severe
  5   Blade_Damage          (bent blade, label=5)
  7   Looseness_Mild
  8   Looseness_Moderate
  9   Looseness_Severe
  10  Bearing_Contamination (sand-in-bearing, outer-race abrasion)
  11  Air_Obstruction       (airflow blocked at fan inlet/outlet)

Feature count: 27  (unchanged from v5)

Dropped (near-zero importance):
  rpm_hz, phase_diff_1x, ay_ax_1x_ratio, az1_ax1_ratio,
  thd_ax, thd_ay, current_A, ay_05x, ax_05x,
  ay_2x_1x, ax_2x_1x, ay_3x_1x, ax_3x_1x, ay_1x, az_1x

Improvements:
  - Normalized spectral entropy (0-1 range) instead of raw nats
  - Absolute value of autocorrelation (magnitude of periodicity, not sign)
  - Fixed 50 Hz everywhere (3000 RPM)

v6 changes (everything else identical to v5):
  - LABEL_NAMES extended with label 10 (Bearing_Contamination)
    and label 11 (Air_Obstruction)
  - DATA_FILES dict extended with the two new CSV filenames
  - Per-class summary now prints all present labels
  - Bearing contamination check block added (mirrors blade-damage check)
  - Air obstruction check block added
"""

import os
import numpy  as np
import pandas as pd
from scipy.stats import kurtosis, skew

# ── Config ─────────────────────────────────────────────────────────────────────
SAVE_DIR    = r'C:\Users\Dell\OneDrive\Documents\Fan_Project'
SAMPLE_RATE = 1000
RPM_HZ      = 50.0

LABEL_NAMES = {
    0:'Normal', 1:'Imbalance_Mild', 2:'Imbalance_Moderate', 3:'Imbalance_Severe',
    5:'Blade_Damage',
    7:'Looseness_Mild', 8:'Looseness_Moderate', 9:'Looseness_Severe',
    11:'Bearing_Contamination',
    13:'Air_Obstruction',
}

# CSV filename → integer label
# Edit paths here if your files live in a sub-folder
DATA_FILES = {
    'Normal.csv':               0,
    'Mass_Imbalance_Mild.csv':  1,
    'Mass_Imbalance_Moderate.csv': 2,
    'Mass_Imbalance_Severe.csv':   3,
    # label 4 reserved / not recorded
    'Blade_Damage.csv':         5,
    # labels 6 reserved / not recorded
    'Looseness_Mild.csv':       7,
    'Looseness_Moderate.csv':   8,
    'Looseness_Severe.csv':     9,
    'Bearing_Moderate.csv':    11,   # ← new  (sand-contaminated bearing)
    'Air_Obstruction.csv':     13,   # ← new  (inlet/outlet blocked)
}

# Normaliser for spectral entropy: log(N/2 + 1) where N=1024
_MAX_ENTROPY = np.log(1024 // 2 + 1)   # log(513) = 6.2403
_LAG_1REV    = int(SAMPLE_RATE / RPM_HZ)       # 20 samples
_LAG_2REV    = int(2 * SAMPLE_RATE / RPM_HZ)   # 40 samples


# ── Helpers ────────────────────────────────────────────────────────────────────

def remove_dc(s: np.ndarray) -> np.ndarray:
    return s - np.mean(s)


def goertzel(sig: np.ndarray, hz: float, fs: int = SAMPLE_RATE) -> float:
    """Single-bin DFT magnitude, normalised by N."""
    n = len(sig); k = n * hz / fs
    omega = 2.0 * np.pi * k / n; coeff = 2.0 * np.cos(omega)
    s0 = s1 = s2 = 0.0
    for x in sig:
        s0 = x + coeff * s1 - s2;  s2 = s1;  s1 = s0
    return np.sqrt(s1*s1 + s2*s2 - coeff*s1*s2) / n


def norm_spectral_entropy(sig: np.ndarray) -> float:
    """
    Spectral entropy normalised to [0, 1].
    0 = all energy at one frequency (pure tone, typical of imbalance on AX)
    1 = perfectly flat spectrum (white noise)
    """
    ps      = np.abs(np.fft.rfft(sig)) ** 2
    ps_norm = ps / (ps.sum() + 1e-30)
    raw     = -np.sum(ps_norm * np.log(ps_norm + 1e-30))
    return float(raw / _MAX_ENTROPY)


def abs_autocorr(sig: np.ndarray, lag: int) -> float:
    """
    |Normalised autocorrelation| at a fixed sample lag.
    Using absolute value captures strength of periodicity regardless of sign.
    lag=20 → 1 revolution at 50 Hz
    lag=40 → 2 revolutions
    """
    s = sig - sig.mean();  n = len(s)
    if lag >= n:
        return 0.0
    a = s[:n - lag];  b = s[lag:]
    denom = np.std(a) * np.std(b)
    return abs(float(np.mean(a * b) / denom)) if denom > 1e-10 else 0.0


# ── Main feature function ──────────────────────────────────────────────────────

def extract_features(ax_raw: np.ndarray,
                     ay_raw: np.ndarray,
                     az_raw: np.ndarray) -> list:
    """
    27 features from one 1024-sample window.
    All harmonic calculations at fixed RPM_HZ = 50.0 Hz.

    Index  Name                   Notes
    -----  --------------------   ------------------------------------------
    0      rms_ax                 AC-RMS on dominant vibration axis
    1      rms_ay                 AC-RMS lateral axis
    2      rms_az                 AC-RMS axial axis
    3      ax_1x                  Goertzel at 50 Hz on AX
    4      ax_2x                  Goertzel at 100 Hz on AX
    5      ax_3x                  Goertzel at 150 Hz on AX
    6      ay_2x                  Goertzel at 100 Hz on AY
    7      ay_3x                  Goertzel at 150 Hz on AY
    8      az_2x                  Goertzel at 100 Hz on AZ
    9      ay_ax_rms_ratio        rms_ay / rms_ax  (imbalance severity)
    10     ax_2x_1x               ax_2x / ax_1x
    11     ax_3x_1x               ax_3x / ax_1x
    12     sent_ax_norm           Normalised spectral entropy AX  [0,1]
    13     sent_ay_norm           Normalised spectral entropy AY  [0,1]
    14     peak_ax
    15     crest_ax               peak / rms
    16     p2p_ax                 peak-to-peak
    17     kurt_ax                kurtosis
    18     skew_ax                skewness
    19     shape_ax               rms / mean_abs
    20     peak_ay
    21     crest_ay
    22     p2p_ay
    23     kurt_ay
    24     az_ax_rms_ratio     ★  rms_az / rms_ax  (primary bent-blade feature)
    25     abs_autocorr_az_1rev★  |autocorr(AZ, lag=20)|
    26     abs_autocorr_az_2rev★  |autocorr(AZ, lag=40)|
    """
    eps = 1e-10

    ax = remove_dc(ax_raw)
    ay = remove_dc(ay_raw)
    az = remove_dc(az_raw)

    rms_ax = float(np.sqrt(np.mean(ax**2)))
    rms_ay = float(np.sqrt(np.mean(ay**2)))
    rms_az = float(np.sqrt(np.mean(az**2)))

    ax_1x  = goertzel(ax, RPM_HZ * 1)
    ax_2x  = goertzel(ax, RPM_HZ * 2)
    ax_3x  = goertzel(ax, RPM_HZ * 3)
    ay_2x  = goertzel(ay, RPM_HZ * 2)
    ay_3x  = goertzel(ay, RPM_HZ * 3)
    az_2x  = goertzel(az, RPM_HZ * 2)

    ay_ax_rms_ratio = rms_ay / (rms_ax + eps)
    ax_2x_1x        = ax_2x / (ax_1x + eps)
    ax_3x_1x        = ax_3x / (ax_1x + eps)

    sent_ax_norm = norm_spectral_entropy(ax)
    sent_ay_norm = norm_spectral_entropy(ay)

    peak_ax  = float(np.max(np.abs(ax)))
    crest_ax = peak_ax / (rms_ax + eps)
    p2p_ax   = float(np.max(ax) - np.min(ax))
    kurt_ax  = float(kurtosis(ax))
    skew_ax  = float(skew(ax))
    mean_abs_ax = float(np.mean(np.abs(ax)))
    shape_ax = rms_ax / (mean_abs_ax + eps)

    peak_ay  = float(np.max(np.abs(ay)))
    crest_ay = peak_ay / (rms_ay + eps)
    p2p_ay   = float(np.max(ay) - np.min(ay))
    kurt_ay  = float(kurtosis(ay))

    # ★ Bent-blade features
    az_ax_rms_ratio      = rms_az / (rms_ax + eps)
    abs_autocorr_az_1rev = abs_autocorr(az, _LAG_1REV)
    abs_autocorr_az_2rev = abs_autocorr(az, _LAG_2REV)

    return [
        rms_ax, rms_ay, rms_az,
        ax_1x, ax_2x, ax_3x,
        ay_2x, ay_3x,
        az_2x,
        ay_ax_rms_ratio,
        ax_2x_1x, ax_3x_1x,
        sent_ax_norm, sent_ay_norm,
        peak_ax, crest_ax, p2p_ax, kurt_ax, skew_ax, shape_ax,
        peak_ay, crest_ay, p2p_ay, kurt_ay,
        az_ax_rms_ratio,
        abs_autocorr_az_1rev,
        abs_autocorr_az_2rev,
    ]


FEAT_NAMES = [
    'rms_ax','rms_ay','rms_az',
    'ax_1x','ax_2x','ax_3x',
    'ay_2x','ay_3x',
    'az_2x',
    'ay_ax_rms_ratio',
    'ax_2x_1x','ax_3x_1x',
    'sent_ax_norm','sent_ay_norm',
    'peak_ax','crest_ax','p2p_ax','kurt_ax','skew_ax','shape_ax',
    'peak_ay','crest_ay','p2p_ay','kurt_ay',
    'az_ax_rms_ratio',
    'abs_autocorr_az_1rev',
    'abs_autocorr_az_2rev',
]

assert len(FEAT_NAMES) == 27, f"Expected 27, got {len(FEAT_NAMES)}"
N_FEATURES = 27


# ── Run ────────────────────────────────────────────────────────────────────────

def main():
    # ── Build combined_dataset.csv from individual per-class CSVs ──────────────
    # This replaces reading a pre-merged file so that adding a new class only
    # requires dropping its CSV into SAVE_DIR and adding one line to DATA_FILES.
    output_path = os.path.join(SAVE_DIR, 'features_dataset.csv')

    ax_cols = [f'ax_{i}' for i in range(1024)]
    ay_cols = [f'ay_{i}' for i in range(1024)]
    az_cols = [f'az_{i}' for i in range(1024)]

    features_list, labels = [], []
    total_windows = 0

    for csv_name, label_id in DATA_FILES.items():
        csv_path = os.path.join(SAVE_DIR, csv_name)
        if not os.path.isfile(csv_path):
            print(f"  [SKIP] {csv_name} not found — label {label_id} "
                  f"({LABEL_NAMES.get(label_id,'?')}) will be absent from dataset.")
            continue

        df_cls = pd.read_csv(csv_path)
        n = len(df_cls)
        print(f"  Loading {csv_name:35s}  label={label_id:2d}  "
              f"({LABEL_NAMES.get(label_id,'?'):22s})  {n:5d} windows")

        for _, row in df_cls.iterrows():
            ax = row[ax_cols].values.astype(np.float64)
            ay = row[ay_cols].values.astype(np.float64)
            az = row[az_cols].values.astype(np.float64)
            features_list.append(extract_features(ax, ay, az))
            labels.append(label_id)

        total_windows += n
        if total_windows % 1000 < n:          # rough progress every ~1000 rows
            print(f"    … {total_windows} windows processed so far")

    if not features_list:
        raise RuntimeError("No data loaded — check SAVE_DIR and DATA_FILES.")

    feat_df          = pd.DataFrame(features_list, columns=FEAT_NAMES)
    feat_df['label'] = labels
    feat_df.to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}  "
          f"({feat_df.shape[0]} rows × {feat_df.shape[1]} cols)")

    # ── Per-class summary ───────────────────────────────────────────────────────
    print(f"\n{'Class':26s}  {'rms_ax':>7} {'ay/ax':>6} {'sent_ax':>7} "
          f"{'sent_ay':>7} ★{'az/ax':>6} ★{'ac1':>7} ★{'ac2':>7}")
    print("-" * 90)
    for lbl in sorted(feat_df['label'].unique()):
        r = feat_df[feat_df['label'] == lbl]
        print(f"  {LABEL_NAMES.get(lbl,str(lbl)):24s}: "
              f"rms_ax={r['rms_ax'].mean():.4f}  "
              f"ay/ax={r['ay_ax_rms_ratio'].mean():.3f}  "
              f"sent_ax={r['sent_ax_norm'].mean():.4f}  "
              f"sent_ay={r['sent_ay_norm'].mean():.4f}  "
              f"★az/ax={r['az_ax_rms_ratio'].mean():.3f}  "
              f"★ac1={r['abs_autocorr_az_1rev'].mean():.4f}  "
              f"★ac2={r['abs_autocorr_az_2rev'].mean():.4f}")

    # ── Blade damage check (unchanged from v5) ──────────────────────────────────
    bd  = feat_df[feat_df['label'] == 5]
    imb = feat_df[feat_df['label'].isin([1, 2, 3])]
    if len(bd) > 0:
        print("\n=== BENT BLADE SEPARATION CHECK ===")
        checks = [
            ('az_ax_rms_ratio',      1.2,  'az/ax > 1.2'),
            ('abs_autocorr_az_1rev', 0.15, '|ac_az1| > 0.15'),
        ]
        for feat, thr, desc in checks:
            bd_pct  = (bd[feat]  > thr).mean() * 100
            imb_pct = (imb[feat] > thr).mean() * 100
            ok = '✓' if bd_pct > 75 and imb_pct < 30 else '✗ check recording'
            print(f"  {desc:30s}: BD={bd_pct:.0f}%  Imb={imb_pct:.0f}%  {ok}")
    else:
        print("\n[Label 5 (Blade_Damage) not found — "
              "run after recording blade damage]")

    # ── Bearing contamination check (new) ──────────────────────────────────────
    brg  = feat_df[feat_df['label'] == 10]
    norm = feat_df[feat_df['label'] == 0]
    if len(brg) > 0:
        print("\n=== BEARING CONTAMINATION SEPARATION CHECK ===")
        # Key discriminator identified in analysis: elevated ax_2x_1x ratio and
        # elevated abs_autocorr_az_1rev relative to Normal/Looseness classes
        loose = feat_df[feat_df['label'].isin([7, 8, 9])]
        checks_brg = [
            ('ax_2x_1x',             3.0,  'ax_2x_1x > 3.0'),
            ('abs_autocorr_az_1rev', 0.35, '|ac_az1| > 0.35'),
        ]
        print(f"  {'Condition':32s} {'Bearing':>10} {'Normal':>10} {'Looseness':>10}")
        for feat, thr, desc in checks_brg:
            brg_pct  = (brg[feat]  > thr).mean() * 100
            norm_pct = (norm[feat] > thr).mean() * 100
            loose_pct = (loose[feat] > thr).mean() * 100 if len(loose) > 0 else float('nan')
            ok = '✓' if brg_pct > 50 and norm_pct < 20 else '✗ check recording'
            print(f"  {desc:32s}  {brg_pct:9.0f}%  {norm_pct:9.0f}%  "
                  f"{loose_pct:9.0f}%  {ok}")
    else:
        print("\n[Label 10 (Bearing_Contamination) not found — "
              "add Bearing_Moderate.csv to SAVE_DIR]")

    # ── Air obstruction check (new) ─────────────────────────────────────────────
    air = feat_df[feat_df['label'] == 11]
    if len(air) > 0:
        print("\n=== AIR OBSTRUCTION SEPARATION CHECK ===")
        # Key discriminators: very high rms_az and both autocorr_az lags
        checks_air = [
            ('rms_az',               0.40, 'rms_az > 0.40'),
            ('abs_autocorr_az_1rev', 0.40, '|ac_az1| > 0.40'),
            ('abs_autocorr_az_2rev', 0.40, '|ac_az2| > 0.40'),
        ]
        print(f"  {'Condition':32s} {'Air_Obs':>10} {'Normal':>10}")
        for feat, thr, desc in checks_air:
            air_pct  = (air[feat]  > thr).mean() * 100
            norm_pct = (norm[feat] > thr).mean() * 100
            ok = '✓' if air_pct > 90 and norm_pct < 10 else '✗ check recording'
            print(f"  {desc:32s}  {air_pct:9.0f}%  {norm_pct:9.0f}%  {ok}")
    else:
        print("\n[Label 11 (Air_Obstruction) not found — "
              "add Air_Obstruction.csv to SAVE_DIR]")


if __name__ == '__main__':
    main()
