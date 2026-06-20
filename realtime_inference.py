"""
realtime_predict.py  v5  —  Real-time Fan Fault Prediction
===========================================================
Rule-based classifier replacing the broken rms_az override.

Rules (calibrated from your real-time measurements):
─────────────────────────────────────────────────────
  R0  rms_ax < 0.030 G                    → FAN OFF
  R1  rms_ax < 0.080 G                    → Normal
  R2  az/ax  < 1.60                       → Imbalance
        rms_az < 0.240                    →   Mild
        rms_az 0.240–0.290                →   Moderate
        rms_az > 0.290                    →   Severe
  R3  az/ax ≥ 1.60  AND  rms_az > 0.215  → Blade Damage
  R4  az/ax ≥ 1.60  AND  rms_az ≤ 0.215  → Looseness
        severity kept from model

Why az/ax separates cleanly (real-time observations):
  Imbalance:     1.1 – 1.6    (tape adds radial, bent blade adds axial)
  Blade Damage:  2.3 – 2.6    (high axial, moderate radial)
  Looseness:     1.7 – 2.5
  Normal:        3.7 – 6.7    (caught earlier by rms_ax < 0.08)

Display: fault name on one line, severity highlighted separately.
ESP32:   receives only class name (no index prefix).

Usage
─────
  python realtime_predict.py --port COM4
  python realtime_predict.py --port COM4 --debug
"""

import argparse, time, sys, os, collections, json
import numpy  as np
import serial
from scipy.stats import kurtosis, skew
import pickle

# ── Paths & constants ──────────────────────────────────────────────────────────
MODEL_DIR   = r'C:\Users\Dell\OneDrive\Documents\Fan_Project'
BAUD_RATE   = 460800
SAMPLE_RATE = 1000
RPM_HZ      = 50.0
WINDOW_SIZE = 1024
N_FEATURES  = 27

_MAX_ENTROPY = np.log(WINDOW_SIZE // 2 + 1)
_LAG_1REV    = int(SAMPLE_RATE / RPM_HZ)   # 20 samples
_LAG_2REV    = int(2 * SAMPLE_RATE / RPM_HZ)  # 40 samples

# ── Calibrated thresholds (G-units) ───────────────────────────────────────────
FAN_OFF_RMS_AX    = 0.030   # below → fan is off (sensor noise only)
NORMAL_RMS_AX     = 0.080   # below → Normal (balanced, quiet radial)
IMBALANCE_AZ_AX   = 1.20    # below → Imbalance (tape dominates radial)
BLADE_RMS_AZ      = 0.145   # above (with high az/ax) → Blade Damage
                             # gap confirmed: Looseness max=0.21, Blade min=0.22
IMB_MILD_MAX      = 0.240   # imbalance rms_az severity boundaries
IMB_MODERATE_MAX  = 0.290


# ── Feature extraction (exact mirror of 02_extract_features.py) ───────────────
def remove_dc(s):  return s - np.mean(s)

def goertzel(sig, hz, fs=SAMPLE_RATE):
    n=len(sig); k=n*hz/fs; omega=2*np.pi*k/n; coeff=2*np.cos(omega)
    s0=s1=s2=0.
    for x in sig: s0=x+coeff*s1-s2; s2=s1; s1=s0
    return np.sqrt(s1*s1+s2*s2-coeff*s1*s2)/n

def norm_spectral_entropy(sig):
    ps=np.abs(np.fft.rfft(sig))**2; pn=ps/(ps.sum()+1e-30)
    return float(-np.sum(pn*np.log(pn+1e-30)))/_MAX_ENTROPY

def abs_autocorr(sig, lag):
    s=sig-sig.mean(); n=len(s)
    if lag>=n: return 0.
    a=s[:n-lag]; b=s[lag:]
    d=np.std(a)*np.std(b)
    return abs(float(np.mean(a*b)/d)) if d>1e-10 else 0.

def extract_features(ax_raw, ay_raw, az_raw):
    eps=1e-10
    ax=remove_dc(ax_raw); ay=remove_dc(ay_raw); az=remove_dc(az_raw)
    rms_ax=float(np.sqrt(np.mean(ax**2)))
    rms_ay=float(np.sqrt(np.mean(ay**2)))
    rms_az=float(np.sqrt(np.mean(az**2)))
    ax_1x=goertzel(ax,RPM_HZ); ax_2x=goertzel(ax,RPM_HZ*2); ax_3x=goertzel(ax,RPM_HZ*3)
    ay_2x=goertzel(ay,RPM_HZ*2); ay_3x=goertzel(ay,RPM_HZ*3)
    az_2x=goertzel(az,RPM_HZ*2)
    ay_ax_r=rms_ay/(rms_ax+eps)
    ax_2_1=ax_2x/(ax_1x+eps); ax_3_1=ax_3x/(ax_1x+eps)
    se_ax=norm_spectral_entropy(ax); se_ay=norm_spectral_entropy(ay)
    p_ax=float(np.max(np.abs(ax))); cr_ax=p_ax/(rms_ax+eps)
    p2p_ax=float(np.max(ax)-np.min(ax))
    k_ax=float(kurtosis(ax)); sk_ax=float(skew(ax))
    sh_ax=rms_ax/(float(np.mean(np.abs(ax)))+eps)
    p_ay=float(np.max(np.abs(ay))); cr_ay=p_ay/(rms_ay+eps)
    p2p_ay=float(np.max(ay)-np.min(ay)); k_ay=float(kurtosis(ay))
    az_ax_r=rms_az/(rms_ax+eps)
    ac1=abs_autocorr(az,_LAG_1REV); ac2=abs_autocorr(az,_LAG_2REV)
    return np.array([
        rms_ax,rms_ay,rms_az, ax_1x,ax_2x,ax_3x, ay_2x,ay_3x, az_2x,
        ay_ax_r, ax_2_1,ax_3_1, se_ax,se_ay,
        p_ax,cr_ax,p2p_ax,k_ax,sk_ax,sh_ax,
        p_ay,cr_ay,p2p_ay,k_ay,
        az_ax_r, ac1, ac2,
    ], dtype=np.float32)

FEAT_NAMES=['rms_ax','rms_ay','rms_az','ax_1x','ax_2x','ax_3x',
            'ay_2x','ay_3x','az_2x','ay_ax_rms_ratio','ax_2x_1x','ax_3x_1x',
            'sent_ax_norm','sent_ay_norm','peak_ax','crest_ax','p2p_ax',
            'kurt_ax','skew_ax','shape_ax','peak_ay','crest_ay','p2p_ay','kurt_ay',
            'az_ax_rms_ratio','abs_autocorr_az_1rev','abs_autocorr_az_2rev']


# ── Rule-based classifier ──────────────────────────────────────────────────────
def classify(feat, model_idx, model_conf, class_names):
    """
    Returns (fault, severity, confidence, source)
    fault    : 'Normal' | 'Imbalance' | 'Blade_Damage' | 'Looseness' | 'FAN_OFF'
    severity : 'Mild' | 'Moderate' | 'Severe' | ''
    source   : 'rule' | 'model'
    """
    rms_ax = float(feat[0])
    rms_az = float(feat[2])
    az_ax  = float(feat[24])

    # R0 — fan off
    if rms_ax < FAN_OFF_RMS_AX:
        return 'FAN_OFF', '', 1.0, 'rule'

    # R1 — normal (very quiet radial axis, fan balanced)
    if rms_ax < NORMAL_RMS_AX:
        return 'Normal', '', model_conf, 'rule'

    # R2 — imbalance (low az/ax: tape creates radial vibration, not axial)
    if az_ax < IMBALANCE_AZ_AX:
        if   rms_az < IMB_MILD_MAX:     sev = 'Mild'
        elif rms_az < IMB_MODERATE_MAX: sev = 'Moderate'
        else:                            sev = 'Severe'
        return 'Imbalance', sev, model_conf, 'rule'

    # R3 — blade damage (high az/ax + elevated rms_az)
    model_name = class_names.get(model_idx, '')

    if 'Bearing_Contamination' in model_name:
        return 'Bearing_Contamination', '', model_conf, 'model'
    
    if 'Air_Obstruction' in model_name:
        return 'Air_Obstruction', '', model_conf, 'model'

    if 'Blade_Damage' in model_name:
        return 'Blade_Damage', '', model_conf, 'model'

    # R4 — looseness (high az/ax + normal rms_az)
    # Use model for severity since model detects looseness correctly
    model_name = class_names.get(model_idx, '')
    if 'Looseness' in model_name:
        sev = model_name.rsplit('_', 1)[-1] if '_' in model_name else ''
        return 'Looseness', sev, model_conf, 'model'

    # Fallback: az/ax is in looseness range but model didn't say Looseness
    # Use az/ax sub-ranges for severity
    if   az_ax < 1.90: sev = 'Mild'
    elif az_ax < 2.20: sev = 'Moderate'
    else:               sev = 'Severe'
    return 'Looseness', sev, model_conf, 'rule'


# ── Load helpers ───────────────────────────────────────────────────────────────
def load_model():
    for fn in ['rf_model.pkl','scaler.pkl']:
        p=os.path.join(MODEL_DIR,fn)
        if not os.path.exists(p): print(f'[ERROR] {p} not found'); sys.exit(1)
    with open(os.path.join(MODEL_DIR,'rf_model.pkl'),'rb') as f: model=pickle.load(f)
    with open(os.path.join(MODEL_DIR,'scaler.pkl'),'rb') as f:   scaler=pickle.load(f)
    print(f'[OK] {type(model).__name__}  |  {len(model.classes_)} classes')
    return model, scaler

def load_class_names():
    p=os.path.join(MODEL_DIR,'class_names.json')
    if os.path.exists(p):
        with open(p) as f: raw=json.load(f)
        names={int(k):v for k,v in raw.items()}
        print(f'[OK] class_names.json loaded: {names}')
        return names
    # Fallback: labels 0,1,2,3,5,7,8,9 → contiguous 0-7
    fallback={0:'Normal',1:'Imbalance_Mild',2:'Imbalance_Moderate',
              3:'Imbalance_Severe',4:'Blade_Damage',
              5:'Looseness_Mild',6:'Looseness_Moderate',7:'Looseness_Severe',
              8:'Bearing_Contamination',9:'Air_Obstruction'}
    print('[WARN] class_names.json not found, using fallback map')
    return fallback

def parse_win(line):
    parts=line.strip().split()
    if not parts or parts[0]!='WIN': return None
    vals=parts[1:]
    if len(vals)!=3*WINDOW_SIZE: return None
    try: d=np.array(vals,dtype=np.float32)
    except ValueError: return None
    return d[:WINDOW_SIZE],d[WINDOW_SIZE:2*WINDOW_SIZE],d[2*WINDOW_SIZE:]


# ── ANSI colour helpers ────────────────────────────────────────────────────────
R='\033[0m'; B='\033[1m'
FAULT_COL = {
    'Normal':       '\033[92m',
    'Imbalance':    '\033[93m',
    'Blade_Damage': '\033[95m',
    'Looseness':    '\033[94m',
    'FAN_OFF':      '\033[90m',
}
SEV_COL = {
    'Mild':     '\033[93m',
    'Moderate': '\033[38;5;208m',
    'Severe':   '\033[91m',
}

def fmt_fault(fault, severity):
    fc  = FAULT_COL.get(fault, '')
    sc  = SEV_COL.get(severity, '')
    name = fault.replace('_',' ')
    if severity:
        return f"{fc}{name}{R} [{sc}{B}{severity}{R}]"
    return f"{fc}{name}{R}"

def fmt_fault_plain(fault, severity):
    """Plain text version for ESP32."""
    name = fault.replace('_',' ')
    return f"{name} {severity}".strip()

def pad(s, n):
    """Pad a string that may contain ANSI codes to visual width n."""
    import re
    visible = re.sub(r'\033\[[0-9;]*m','',s)
    return s + ' '*(max(0, n-len(visible)))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--port',    default='COM4')
    ap.add_argument('--baud',    type=int,   default=BAUD_RATE)
    ap.add_argument('--timeout', type=float, default=2.0)
    ap.add_argument('--debug',   action='store_true',
                    help='Print all 27 feature values per window')
    args=ap.parse_args()

    model, scaler = load_model()
    CLASS_NAMES   = load_class_names()

    print(f'\n[CONFIG] Port={args.port}  Baud={args.baud}  Units=G')
    print(f'[CONFIG] Thresholds:')
    print(f'         Fan OFF   : rms_ax < {FAN_OFF_RMS_AX:.3f} G')
    print(f'         Normal    : rms_ax < {NORMAL_RMS_AX:.3f} G')
    print(f'         Imbalance : az/ax  < {IMBALANCE_AZ_AX:.2f}')
    print(f'         Blade Dmg : rms_az > {BLADE_RMS_AZ:.3f} G  (when az/ax ≥ {IMBALANCE_AZ_AX:.2f})')
    print()

    # Header
    h1 = f"{'Win':>4}  {'Fault [Severity]':32}  {'Conf':>5}  {'rms_az':>7}  {'az/ax':>6}  {'rms_ax':>7}  {'Src':>5}"
    print(h1)
    print('─'*len(h1))

    history   = collections.deque(maxlen=10)
    win_count = 0
    t_start   = time.time()

    try: ser=serial.Serial(args.port,args.baud,timeout=args.timeout)
    except serial.SerialException as e: print(f'[ERROR] {e}'); sys.exit(1)
    time.sleep(2); ser.reset_input_buffer()

    try:
        while True:
            raw=ser.readline()
            if not raw: continue
            try: line=raw.decode('utf-8',errors='replace')
            except: continue
            if line.startswith('#') or not line.startswith('WIN'): continue

            result=parse_win(line)
            if result is None: continue
            ax,ay,az=result

            t0=time.perf_counter()
            feat=extract_features(ax,ay,az)
            t_ms=(time.perf_counter()-t0)*1000

            feat_sc   =scaler.transform(feat.reshape(1,-1))
            model_idx =int(model.predict(feat_sc)[0])
            proba     =model.predict_proba(feat_sc)[0]
            model_conf=float(np.max(proba))

            fault,severity,conf,src=classify(feat,model_idx,model_conf,CLASS_NAMES)

            # Send to ESP32 — only class name, no index prefix
            esp_msg = fmt_fault_plain(fault, severity)
            ser.write(f'{esp_msg}\n'.encode('utf-8'))

            win_count+=1
            history.append(f'{fault}_{severity}' if severity else fault)
            rms_ax=float(feat[0]); rms_az=float(feat[2]); az_ax=float(feat[24])

            fstr = fmt_fault(fault, severity)
            src_col = '\033[90m' if src=='rule' else '\033[36m'
            print(f"{win_count:>4}  {pad(fstr,32)}  "
                  f"{conf:5.1%}  {rms_az:7.4f}  {az_ax:6.3f}  {rms_ax:7.4f}  "
                  f"{src_col}{src:>5}{R}  [{t_ms:.1f}ms]")

            if args.debug:
                for i,(n,v) in enumerate(zip(FEAT_NAMES,feat)):
                    print(f"      [{i:2d}] {n:28s}= {v:.6f}")

            if win_count%5==0:
                counts  = collections.Counter(history)
                maj_key = counts.most_common(1)[0][0]
                parts   = maj_key.rsplit('_',1)
                mf,ms   = (parts[0],parts[1]) if (len(parts)==2 and parts[1] in ('Mild','Moderate','Severe')) else (maj_key,'')
                print(f"      {'─'*60}")
                print(f"      {B}Vote (last 5): {pad(fmt_fault(mf,ms),0)}{R}   "
                      f"rms_az={rms_az:.4f} G  az/ax={az_ax:.3f}")
                print(f"      {'─'*60}")

    except KeyboardInterrupt:
        print(f'\n[INFO] Stopped. Windows: {win_count}')
    finally:
        ser.close()

if __name__=='__main__':
    main()