"""
=============================================================================
  PHYSICS-INFORMED FEATURE ENGINEERING PIPELINE
  Jet Impingement CFD → SciML Dataset

  CRITICAL FIXES FROM DATA INSPECTION:
    FIX A : domain_y_max corrected to 0.027 (was 0.02). The actual mesh goes
             to Y=0.027. The inlet is at y≈0.027, not y=0.02. This was making
             signed_dist_wall and inlet detection both wrong.

    FIX B : Sampling is now DETERMINISTIC GEOMETRIC (tier-based, seed=42):
             Tier 0 (chip wall, y < 0.0001): ALL nodes — the hottest zone.
                99.4% of nodes are ambient; wall nodes MUST be force-included
                or the model never sees a hot training example.
             Tier 1 (jet core, r < D_m/2 AND y > y_max-5mm): ALL nodes — inlet.
             Tier 2 (rest): fixed uniform seed=42, NOT physics-weighted.
             This is IDENTICAL to what dimen_red.py uses. The graph template
             node positions now match the feature file node positions exactly.
             Previous mismatch: dimen_red used max-disturbance prob sampling
             (seed=42, one reference file); FE used tier-0 + per-file-seeded
             prob sampling. Features at index i described a different physical
             location than the graph edge at index i — explaining all failures.

    FIX C : bc_velocity normalization: changed from (vel-VEL_MIN)/(VEL_MAX-VEL_MIN)
             to vel/VEL_MAX. The old formula gave bc_velocity=0 for the minimum
             inlet velocity (3 m/s), removing the feature signal entirely for
             the slowest (and hottest) simulations.

    FIX D : signed_dist_wall now correctly uses domain_y_max=0.027.
             Formula: domain_y_max - ys (0 at top/inlet, positive downward).
             This correctly encodes distance from the inlet plane.

  Architecture:
    Inputs  (16): Log_Reynolds, Heat_Flux, Stanton_Proxy,       ← Global (3)
                  X, Y, Z, Radius, Dist_Outflow, Dist_Outlet,
                  signed_dist_wall                               ← Spatial (7)
                  Y_norm,                                        ← Skewed (1)
                  Stagnation_Flag, is_HD4, is_HD6, Re_regime,
                  bc_velocity                                    ← Binary (5)
    Targets  (5): Temperature, Pressure, U_vel, V_vel, W_vel
    Aux      (2): Theta_norm, Nu_local
=============================================================================
"""

import os
import gc
import glob
import re
import hashlib
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import StratifiedShuffleSplit
import joblib
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────
#  DOMAIN CONSTANTS
# ─────────────────────────────────────────────────
FLUID = dict(rho=998.0, mu=0.001003, k=0.6, Pr=7.01, cp=4182.0)
GEOM  = dict(
    D_m          = 0.00275,
    A_chip_m2    = 30.25e-6,
    jet_center_x = 0.05238,
    jet_center_z = 0.012,
    domain_x_max = 0.10,
    domain_y_max = 0.027,      # FIX A: was 0.02, actual mesh goes to 0.027
    domain_z_max = 0.024,
    wall_y_thresh= 0.005,
    T_inlet      = 20.0,
    jet_y_min    = 0.005,      # jet core: y > domain_y_max - jet_y_min
)
SAMPLE = dict(points_per_sim = 250_000, seed = 42)
RE_REGIME_THRESHOLD = 15_000
VEL_MAX = 12.0   # FIX C: only need VEL_MAX now; normalization is vel/VEL_MAX

# Feature layout: 3G + 7S + 1K + 5B = 16
GLOBAL_COLS  = [
    'Log_Reynolds',
    'Heat_Flux',
    'Stanton_Proxy',
    'temp_gradient_proxy'   # 🔥 added
]
SPATIAL_COLS = [
    'X', 'Y', 'Z',
    'Radius', 'inv_radius',   # 🔥 added
    'Dist_Outflow', 'Dist_Outlet',
    'signed_dist_wall'
]
SKEWED_COLS  = ['Y_norm']
BINARY_COLS  = ['Stagnation_Flag', 'is_HD4', 'is_HD6', 'Re_regime', 'bc_velocity']
ALL_INPUT_COLS = GLOBAL_COLS + SPATIAL_COLS + SKEWED_COLS + BINARY_COLS  # 16

# Index map: [0-2 Global | 3-9 Spatial | 10 Skewed | 11-15 Binary]
STAGNATION_FLAG_IDX = 11
BC_VELOCITY_IDX     = 15

TARGET_POWER_COLS = ['Pressure']
TARGET_COLS       = ['Temperature', 'Pressure', 'U_vel', 'V_vel', 'W_vel']
AUX_COLS          = ['Theta_norm', 'Nu_local']


# ─────────────────────────────────────────────────
#  DETERMINISTIC GEOMETRIC NODE SELECTION  (FIX B)
# ─────────────────────────────────────────────────
def select_nodes_deterministic(coords: np.ndarray, target_n: int = 250_000, seed: int = 42) -> np.ndarray:
    """
    Selects a FIXED set of node indices using purely geometric criteria.
    This function MUST produce identical output for any simulation that shares
    the same mesh (same coordinates), so the graph template built in dimen_red.py
    is valid for every training/val/test simulation.

    Tier 0 — chip wall (y ≈ 0, y < 0.0001 m):
        ALL nodes forced. The chip is the heat source; these are the hottest nodes.
        99.4% of domain is ambient, so without force-inclusion the model sees
        almost no hot training examples.

    Tier 1 — jet nozzle inlet (r < D_m/2 AND y > domain_y_max - jet_y_min):
        ALL nodes forced. The inlet carries the velocity BC.
        Without these nodes the model cannot learn the velocity field.

    Tier 2 — everything else:
        Fixed uniform random with seed=42. NOT physics-weighted (physics varies
        per simulation, making it impossible to use the same sampling across sims).
    """
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    y_actual_max = y.max()

    # Tier 0: chip wall
    t0_mask = y < 1e-4
    t0_idx  = np.where(t0_mask)[0]

    # Tier 1: jet nozzle inlet
    radius_from_jet = np.sqrt((x - GEOM['jet_center_x'])**2 + (z - GEOM['jet_center_z'])**2)
    t1_mask = (~t0_mask) & (radius_from_jet < GEOM['D_m'] / 2.0) & (y > y_actual_max - GEOM['jet_y_min'])
    t1_idx  = np.where(t1_mask)[0]

    forced = np.concatenate([t0_idx, t1_idx])
    n_forced = len(forced)

    # Tier 2: remaining budget — uniform random
    rest_mask = ~(t0_mask | t1_mask)
    rest_idx  = np.where(rest_mask)[0]
    budget    = max(target_n - n_forced, 0)
    budget    = min(budget, len(rest_idx))

    rng = np.random.default_rng(seed)
    chosen_rest = rng.choice(rest_idx, size=budget, replace=False)

    selected = np.sort(np.concatenate([forced, chosen_rest]))
    return selected


# ─────────────────────────────────────────────────
#  PRESSURE TRANSFORM HELPERS
# ─────────────────────────────────────────────────
def log_transform_pressure(p: np.ndarray) -> np.ndarray:
    """sign(p) * log1p(|p|) — handles negative gauge pressure, linear inverse."""
    return np.sign(p) * np.log1p(np.abs(p))


def inverse_log_transform_pressure(p_log: np.ndarray) -> np.ndarray:
    return np.sign(p_log) * np.expm1(np.abs(p_log))


# ─────────────────────────────────────────────────
#  PHASE 0: FILE VALIDATION
# ─────────────────────────────────────────────────
def validate_files(folder_path="D:/data/JET/**/*.h5"):
    h5_files = glob.glob(folder_path, recursive=True)
    print(f"[Validate] Found {len(h5_files)} h5 files on disk.")
    valid, skipped = [], []
    for fp in h5_files:
        hd  = re.search(r'[/\\]H(\d+)[/\\]', fp)
        vel = re.search(r'Vel_([-\d\.]+)',     fp)
        pw  = re.search(r'Pow_(\d+)',          fp)
        if not (hd and vel and pw):
            skipped.append((fp, "regex parse failure")); continue
        try:
            with h5py.File(fp, 'r') as f:
                gname   = list(f.keys())[0]
                missing = {'Coordinates', 'Temperature', 'Pressure', 'Velocity'} - set(f[gname].keys())
                if missing:
                    skipped.append((fp, f"missing: {missing}")); continue
        except Exception as e:
            skipped.append((fp, str(e))); continue
        valid.append(fp)
    if skipped:
        print(f"\n[Validate] ⚠  Skipped {len(skipped)} files:")
        for fp, reason in skipped:
            print(f"  {os.path.basename(fp):50s} → {reason}")
    print(f"[Validate] ✓  {len(valid)} files cleared.\n")
    return sorted(valid)


# ─────────────────────────────────────────────────
#  PHASE 1: FEATURE ENGINEERING (per simulation)
# ─────────────────────────────────────────────────
def engineer_one_sim(file_path: str, t_max_global: float = 100.0) -> pd.DataFrame:
    """
    Uses deterministic geometric node selection (FIX B) so every simulation
    provides features at the SAME spatial positions as the graph template.
    """
    hd    = float(re.search(r'[/\\]H(\d+)[/\\]', file_path).group(1))
    vel   = float(re.search(r'Vel_([-\d\.]+)', file_path).group(1))
    pow_w = float(re.search(r'Pow_(\d+)',      file_path).group(1))

    re_num    = (FLUID['rho'] * abs(vel) * GEOM['D_m']) / FLUID['mu']
    heat_flux = pow_w / GEOM['A_chip_m2']
    dT_ref    = max(t_max_global - GEOM['T_inlet'], 1.0)
    q_star    = heat_flux / (FLUID['rho'] * FLUID['cp'] * max(abs(vel), 1e-6) * dT_ref)

    with h5py.File(file_path, 'r') as f:
        gname     = list(f.keys())[0]
        coords    = f[gname]['Coordinates'][:]
        temp      = f[gname]['Temperature'][:].ravel()
        press     = f[gname]['Pressure'][:].ravel()
        vel_field = f[gname]['Velocity'][:]

    # FIX B: deterministic geometric selection — same nodes for every simulation
    sel = select_nodes_deterministic(coords, target_n=SAMPLE['points_per_sim'])

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    xs, ys, zs = x[sel], y[sel], z[sel]
    ts, ps     = temp[sel], press[sel]
    us, vs, ws = vel_field[sel, 0], vel_field[sel, 1], vel_field[sel, 2]

    radius_full = np.sqrt((x - GEOM['jet_center_x'])**2 + (z - GEOM['jet_center_z'])**2)
    radius = radius_full[sel]
    # 🔥 NEW: gradient proxy (now radius is defined)
    temp_gradient_proxy = heat_flux / (radius + 1e-4)

    # 🔥 NEW: inverse radius (makes peak easier to learn)
    inv_radius = 1.0 / (radius + 1e-5)

    dist_outflow = np.minimum.reduce([xs, GEOM['domain_x_max'] - xs,
                                      zs, GEOM['domain_z_max'] - zs])
    outlet_centers = np.array([[0.0, GEOM['jet_center_z']], [GEOM['domain_x_max'], GEOM['jet_center_z']]])
    dist_outlet    = np.min([np.sqrt((xs - cx)**2 + (zs - cz)**2)
                             for cx, cz in outlet_centers], axis=0)

    delta_bl = GEOM['D_m'] / np.sqrt(max(re_num, 1.0))
    # 🔥 Log scaling stabilizes boundary layer representation
    y_norm = np.log1p(ys / (delta_bl + 1e-6))

    # FIX D: signed_dist_wall = distance from inlet (top), using corrected domain_y_max
    # 0 at inlet (y=domain_y_max), positive downward toward chip (y=0)
    signed_dist_wall = (GEOM['domain_y_max'] - ys).astype(np.float32)

    # 🔥 Normalize it
    signed_dist_wall = signed_dist_wall / GEOM['domain_y_max']

    # 🔥 Smooth Gaussian stagnation encoding
    stagnation_flag = np.exp(- (radius / GEOM['D_m'])**2).astype(np.float32)
    is_HD4          = np.float32(hd == 4.0) * np.ones(len(xs), dtype=np.float32)
    is_HD6          = np.float32(hd == 6.0) * np.ones(len(xs), dtype=np.float32)
    re_regime       = (float(re_num) > RE_REGIME_THRESHOLD) * np.ones(len(xs), dtype=np.float32)

    # FIX C: bc_velocity uses vel/VEL_MAX (not (vel-VEL_MIN)/(VEL_MAX-VEL_MIN))
    # Old formula gave bc_velocity=0 for vel=3 m/s (minimum case, hottest sims)
    # New formula: vel=3 → 0.25, vel=12 → 1.0. Always has a signal.
    y_actual_max    = ys.max()
    inlet_node_mask = (ys > y_actual_max - 0.002) & (radius < GEOM['D_m'] / 2.0)
    vel_norm        = abs(vel) / VEL_MAX   # [0.25, 1.0] range
    # 🔥 smoother spatial decay instead of hard cutoff
    bc_velocity = vel_norm * np.exp(-radius / GEOM['D_m'])
    bc_velocity = bc_velocity.astype(np.float32)

    dT_ref_val = max(t_max_global - GEOM['T_inlet'], 1e-6)
    theta_norm = (ts - GEOM['T_inlet']) / dT_ref_val

    wall_sel = ys < 1e-4
    nu_local = np.zeros_like(ts)
    dT_wall  = ts[wall_sel] - GEOM['T_inlet']
    dT_wall  = np.where(np.abs(dT_wall) < 1e-6, 1e-6, dT_wall)
    nu_local[wall_sel] = (heat_flux / dT_wall) * GEOM['D_m'] / FLUID['k']

    n_wall  = wall_sel.sum()
    n_inlet = inlet_node_mask.sum()
    n_hot   = (ts > GEOM['T_inlet'] + 1.0).sum()
    if n_wall == 0:
        print(f"  ⚠ WARNING: No wall nodes selected for {os.path.basename(file_path)}")
    if n_inlet == 0:
        print(f"  ⚠ WARNING: No inlet nodes selected for {os.path.basename(file_path)}")

    return pd.DataFrame({
        'Log_Reynolds'    : np.log(max(re_num, 1.0)),
        'Heat_Flux'       : heat_flux,
        'Stanton_Proxy'   : q_star,
        'temp_gradient_proxy': temp_gradient_proxy,   # 🔥 NEW

        'X': xs, 'Y': ys, 'Z': zs,
        'Radius'          : radius,
        'inv_radius'      : inv_radius,              # 🔥 NEW
        'Dist_Outflow'    : dist_outflow,
        'Dist_Outlet'     : dist_outlet,
        'signed_dist_wall': signed_dist_wall,

        'Y_norm'          : y_norm,

        'Stagnation_Flag' : stagnation_flag,
        'is_HD4'          : is_HD4,
        'is_HD6'          : is_HD6,
        'Re_regime'       : re_regime,
        'bc_velocity'     : bc_velocity,

        'sim_id'          : os.path.basename(file_path),

        'Temperature'     : ts,
        'Pressure'        : ps,
        'U_vel'           : us,
        'V_vel'           : vs,
        'W_vel'           : ws,

        'Theta_norm'      : theta_norm,
        'Nu_local'        : nu_local,
    }), n_wall, n_inlet, n_hot


# ─────────────────────────────────────────────────
#  PHASE 2: EXTRACTION LOOP
# ─────────────────────────────────────────────────
def _file_seed(fp: str) -> int:
    return int(hashlib.md5(os.path.basename(fp).encode()).hexdigest()[:8], 16)


def extract_all(valid_files, t_max_global=None) -> pd.DataFrame:
    if t_max_global is None:
        t_max_global = -np.inf
        for fp in valid_files:
            with h5py.File(fp, 'r') as f:
                gname = list(f.keys())[0]
                t_max_global = max(t_max_global, f[gname]['Temperature'][:].max())
        t_max_global = float(t_max_global)

    chunks = []
    for fp in tqdm(valid_files, desc="Engineering Features"):
        try:
            df_sim, nw, ni, nh = engineer_one_sim(fp, t_max_global=t_max_global)
            chunks.append(df_sim)
            tqdm.write(f"{os.path.basename(fp)} | wall={nw:,} inlet={ni:,} hot={nh:,}")
        except Exception as e:
            print(f"[SKIP] {os.path.basename(fp)}: {e}")
        gc.collect()

    master = pd.concat(chunks, ignore_index=True)
    print(f"\n[Extract] ✓  {len(master):,} rows × {master.shape[1]} cols")
    return master


# ─────────────────────────────────────────────────
#  PHASE 3: SCALING
# ─────────────────────────────────────────────────
def fit_and_transform(master: pd.DataFrame, save_dir: str):
    print("\n[Scale]  Fitting scalers...")

    scaler_global   = StandardScaler()
    scaler_spatial  = StandardScaler()
    scaler_skewed   = PowerTransformer(method='yeo-johnson', standardize=True)
    scaler_temp     = StandardScaler()
    scaler_vel      = StandardScaler()
    scaler_pressure = StandardScaler()
    scaler_aux      = StandardScaler()

    g_scaled = scaler_global .fit_transform(master[GLOBAL_COLS ])
    s_scaled = scaler_spatial.fit_transform(master[SPATIAL_COLS])
    k_scaled = scaler_skewed .fit_transform(master[SKEWED_COLS ])
    b_raw    = master[BINARY_COLS].values.astype(np.float32)

    p_log    = log_transform_pressure(master[TARGET_POWER_COLS].values)
    p_scaled = scaler_pressure.fit_transform(p_log)

    t_scaled = np.concatenate([
        scaler_temp.fit_transform(master[['Temperature']]),
        p_scaled,
        scaler_vel.fit_transform(master[['U_vel', 'V_vel', 'W_vel']]),
    ], axis=1)

    a_scaled = scaler_aux.fit_transform(master[AUX_COLS])

    os.makedirs(save_dir, exist_ok=True)
    for obj, name in [
        (scaler_global,   'scaler_global.pkl'),
        (scaler_spatial,  'scaler_spatial.pkl'),
        (scaler_skewed,   'scaler_skewed.pkl'),
        (scaler_temp,     'scaler_temp.pkl'),
        (scaler_vel,      'scaler_vel.pkl'),
        (scaler_pressure, 'scaler_pressure.pkl'),
        (scaler_aux,      'scaler_aux.pkl'),
    ]:
        joblib.dump(obj, os.path.join(save_dir, name))
    print("[Scale]  ✓  Scalers saved.")

    X_full = np.concatenate([g_scaled, s_scaled, k_scaled, b_raw], axis=1).astype(np.float32)
    T_full = t_scaled.astype(np.float32)
    A_full = a_scaled.astype(np.float32)

    assert X_full.shape[1] == 18, f"Expected 16 features, got {X_full.shape[1]}"

    # Sampling audit
    wall_frac = (master['Y'] < 1e-4).mean()
    hot_frac  = (master['Temperature'] > GEOM['T_inlet'] + 1.0).mean()
    print(f"[Audit]  Wall nodes (y<0.1mm)  : {wall_frac:.2%}")
    print(f"[Audit]  Hot nodes (T>T_in+1°C): {hot_frac:.2%}")
    if hot_frac < 0.005:
        print("[Audit]  ⚠ WARNING: hot node fraction is very low. "
              "Check tier-0 wall selection is working.")

    stats = {
        'mean_T':  float(T_full[:, 0].mean()),
        'std_T':   max(float(T_full[:, 0].std()), 1e-8),
        'mean_P':  float(T_full[:, 1].mean()),
        'std_P':   max(float(T_full[:, 1].std()), 1e-8),
        'mean_Vx': float(T_full[:, 2].mean()),
        'std_Vx':  max(float(T_full[:, 2].std()), 1e-8),
        'mean_Vy': float(T_full[:, 3].mean()),
        'std_Vy':  max(float(T_full[:, 3].std()), 1e-8),
        'mean_Vz': float(T_full[:, 4].mean()),
        'std_Vz':  max(float(T_full[:, 4].std()), 1e-8),
        'outlet_P_bc_target': float((0.0 - scaler_pressure.mean_[0]) / scaler_pressure.scale_[0]),
    }
    torch.save(stats, os.path.join(save_dir, 'training_stats.pt'))
    print(f"[Scale]  ✓  training_stats.pt saved.")
    print(f"           outlet BC target = {stats['outlet_P_bc_target']:.4f}")
    print(f"[Scale]  Input  : {X_full.shape}  (3G+7S+1K+5B = 16)")
    print(f"[Scale]  Target : {T_full.shape}")
    return X_full, T_full, A_full


def transform_only(df: pd.DataFrame, scalers: dict):
    g  = scalers['global'].transform(df[GLOBAL_COLS])
    s  = scalers['spatial'].transform(df[SPATIAL_COLS])
    k  = scalers['skewed'].transform(df[SKEWED_COLS])
    b  = df[BINARY_COLS].values.astype(np.float32)
    t1 = scalers['temp'].transform(df[['Temperature']])
    p_log = log_transform_pressure(df[TARGET_POWER_COLS].values)
    tp = scalers['pressure'].transform(p_log)
    t2 = scalers['vel'].transform(df[['U_vel', 'V_vel', 'W_vel']])
    a  = scalers['aux'].transform(df[AUX_COLS])
    X = np.concatenate([g, s, k, b], axis=1).astype(np.float32)
    T = np.concatenate([t1, tp, t2], axis=1).astype(np.float32)
    A = a.astype(np.float32)
    return X, T, A


# ─────────────────────────────────────────────────
#  PHASE 4: DATASET
# ─────────────────────────────────────────────────
class JetImpingementDataset(Dataset):
    def __init__(self, X: np.ndarray, T: np.ndarray, A: np.ndarray,
                 nodes_per_sim: int = SAMPLE['points_per_sim']):
        num_sims = max(X.shape[0] // nodes_per_sim, 1)
        self.X = torch.from_numpy(X).view(num_sims, nodes_per_sim, X.shape[1])
        self.T = torch.from_numpy(T).view(num_sims, nodes_per_sim, T.shape[1])
        self.A = torch.from_numpy(A).view(num_sims, nodes_per_sim, A.shape[1])

    def __len__(self):             return self.X.shape[0]
    def __getitem__(self, idx):    return self.X[idx], self.T[idx], self.A[idx]
    @property
    def input_dim(self):           return self.X.shape[2]


# ─────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  JET IMPINGEMENT  ·  FEATURE ENGINEERING PIPELINE")
    print("="*60)

    DATA_DIR = r"D:\data\JET"
    all_valid_files = validate_files(folder_path="D:/data/JET/**/*.h5")

    hd_list = [float(re.search(r'[/\\]H(\d+)[/\\]', f).group(1)) for f in all_valid_files]
    re_list = [(FLUID['rho'] * float(re.search(r'Vel_([-\d\.]+)', f).group(1)) * GEOM['D_m']) / FLUID['mu']
               for f in all_valid_files]
    labels  = [f"{hd}_{int(abs(r)//10000)}" for hd, r in zip(hd_list, re_list)]

    sss  = StratifiedShuffleSplit(n_splits=1, test_size=0.15,  random_state=42)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.176, random_state=42)
    train_val_idx, test_idx = next(sss.split(all_valid_files, labels))
    train_idx, val_idx = next(sss2.split(
        np.array(all_valid_files)[train_val_idx], np.array(labels)[train_val_idx]))

    train_files = [all_valid_files[i] for i in train_val_idx[train_idx]]
    val_files   = [all_valid_files[i] for i in train_val_idx[val_idx]]
    test_files  = [all_valid_files[i] for i in test_idx]

    t_max_global = -np.inf
    print("Calculating global T_max from train set...")
    for fp in train_files:
        with h5py.File(fp, 'r') as f:
            gname = list(f.keys())[0]
            t_max_global = max(t_max_global, f[gname]['Temperature'][:].max())
    t_max_global = float(t_max_global)
    np.save(os.path.join(DATA_DIR, 't_max_global.npy'), np.array([t_max_global]))
    print(f"  t_max_global = {t_max_global:.2f} °C")

    print("\n--- Pass 1: Fitting scalers ---")
    df_train_full = extract_all(train_files, t_max_global=t_max_global)
    fit_and_transform(df_train_full, save_dir=DATA_DIR)

    scalers = {k: joblib.load(os.path.join(DATA_DIR, f'scaler_{k}.pkl'))
               for k in ['global', 'spatial', 'skewed', 'temp', 'vel', 'pressure', 'aux']}
    del df_train_full; gc.collect()

    def process_split(split_files, split_name):
        sim_dir = os.path.join(DATA_DIR, f'{split_name}_sims')
        os.makedirs(sim_dir, exist_ok=True)
        for fp in tqdm(split_files, desc=f"  {split_name}"):
            try:
                df_sim, _, _, _ = engineer_one_sim(fp, t_max_global=t_max_global)
                X, T, A = transform_only(df_sim, scalers)
                ds = JetImpingementDataset(X, T, A)
                name = (os.path.basename(os.path.dirname(fp)) + '_' +
                        os.path.basename(fp).replace('.h5', ''))
                torch.save(ds, os.path.join(sim_dir, f'{name}.pt'))
                del df_sim, X, T, A, ds
            except Exception as e:
                print(f"[SKIP] {os.path.basename(fp)}: {e}")
            gc.collect()

    process_split(train_files, 'train')
    process_split(val_files,   'val')
    process_split(test_files,  'test')
    print(f"\n{'='*60}\n  PIPELINE COMPLETE — 16 features, deterministic sampling\n{'='*60}\n")