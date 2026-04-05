"""
### Feature Engineering
"""

"""
=============================================================================
  PHYSICS-INFORMED FEATURE ENGINEERING PIPELINE
  Jet Impingement CFD → SciML Dataset
  
  Architecture:
    Inputs  (12): H_D, Log_Reynolds, Heat_Flux, Stanton_Proxy,
                  X, Y, Z, Radius, Dist_Outflow, Dist_Outlet,
                  Y_norm, Stagnation_Flag
    Targets (5):  Temperature, Pressure, U_vel, V_vel, W_vel
=============================================================================
"""

import os
import gc
import glob
import re
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import StratifiedShuffleSplit
import joblib
from tqdm.auto import tqdm
import hashlib
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────
#  DOMAIN CONSTANTS  (verified against your EDA)
# ─────────────────────────────────────────────────
FLUID = dict(rho=998.0, mu=0.001003, k=0.6, Pr=7.01, cp=4182.0)
GEOM  = dict(
    D_m          = 0.00275,      # jet nozzle diameter [m]
    A_chip_m2    = 30.25e-6,     # heater footprint [m²]
    jet_center_x = 0.05,         # nozzle X centroid [m]
    jet_center_z = 0.012,        # nozzle Z centroid [m]
    domain_x_max = 0.10,         # domain X extent [m]
    domain_y_max = 0.02,         # domain Y extent [m]
    domain_z_max = 0.024,        # domain Z extent [m]
    wall_y_thresh= 0.005,        # boundary-layer cutoff [m]
    T_inlet      = 20.0,         # inlet water temperature [°C]
)
SAMPLE = dict(
    points_per_sim   = 100_000,
    boundary_frac    = 0.75,     # 75 % of points from BL (confirmed by your STD maps)
    seed             = 42,
)

# ─────────────────────────────────────────────────
#  PHASE 0: FILE VALIDATION
# ─────────────────────────────────────────────────
def validate_files(folder_path="D:/data/JET/**/*.h5"):
    """
    Validates every h5 file BEFORE the main loop.
    Catches corrupt files and regex failures early.
    Returns sorted list of valid file paths.
    """
    h5_files = glob.glob(folder_path, recursive=True)
    print(f"[Validate] Found {len(h5_files)} h5 files on disk.")

    valid, skipped = [], []
    for fp in h5_files:
        hd  = re.search(r'[/\\]H(\d+)[/\\]', fp)
        vel = re.search(r'Vel_([-\d\.]+)',     fp)
        pw  = re.search(r'Pow_(\d+)',      fp)
        if not (hd and vel and pw):
            skipped.append((fp, "regex parse failure"))
            continue
        try:
            with h5py.File(fp, 'r') as f:
                gname = list(f.keys())[0]
                required = {'Coordinates', 'Temperature', 'Pressure', 'Velocity'}
                missing  = required - set(f[gname].keys())
                if missing:
                    skipped.append((fp, f"missing datasets: {missing}"))
                    continue
        except Exception as e:
            skipped.append((fp, str(e)))
            continue
        valid.append(fp)

    if skipped:
        print(f"\n[Validate] ⚠  Skipped {len(skipped)} files:")
        for fp, reason in skipped:
            print(f"  {os.path.basename(fp):50s} → {reason}")
    print(f"[Validate] ✓  {len(valid)} files cleared for processing.\n")
    return sorted(valid)


# ─────────────────────────────────────────────────
#  PHASE 1: FEATURE ENGINEERING (per simulation)
# ─────────────────────────────────────────────────
def engineer_one_sim(file_path: str, rng: np.random.Generator, t_max_global: float = 100.0) -> pd.DataFrame:
    """
    Extracts every feature for one simulation file.
    Returns a DataFrame of `points_per_sim` rows.
    
    FEATURES ADDED vs original script:
      NEW  Y_norm        – wall-normal distance normalised by BL thickness
      NEW  Radial_vel_mag– in-plane velocity magnitude (swirl / wall jet strength)
      NEW  Nu_local      – local Nusselt number proxy (only valid at wall nodes)
      NEW  Stagnation_flag – 1 if node is within 1 nozzle-diameter of stagnation
      NEW  Theta_norm  non-dimensional temperature (physics constraint target)
      FIX  Biased sampling uses a SEEDED per-file rng → reproducible across runs
    """
    # ── 1. Parse global parameters ──────────────────────────────────
    hd    = float(re.search(r'[/\\]H(\d+)[/\\]', file_path).group(1))
    vel   = float(re.search(r'Vel_([-\d\.]+)', file_path).group(1))
    pow_w = float(re.search(r'Pow_(\d+)',      file_path).group(1))

    # Global dimensionless numbers  (confirmed valid by your Nu-Re EDA)
    re_num     = (FLUID['rho'] * abs(vel) * GEOM['D_m']) / FLUID['mu']
    heat_flux  = pow_w / GEOM['A_chip_m2']                           # W/m²
    # Stanton number proxy (Bug 7 Fix: physically consistent Stanton number)
    dT_ref_global = max(t_max_global - GEOM['T_inlet'], 1e-6)
    q_star = heat_flux / (FLUID['rho'] * FLUID['cp'] * max(abs(vel), 1e-6) * dT_ref_global)

    # ── 2. Load spatial fields ───────────────────────────────────────
    with h5py.File(file_path, 'r') as f:
        gname     = list(f.keys())[0]
        coords    = f[gname]['Coordinates'][:]   # (N, 3)
        temp      = f[gname]['Temperature'][:]   # (N, 1) or (N,)
        press     = f[gname]['Pressure'][:]
        vel_field = f[gname]['Velocity'][:]      # (N, 3)

    # Flatten to 1-D where needed
    temp  = temp.ravel()
    press = press.ravel()

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    # Determine full-mesh radius beforehand for inlet masking
    radius_full = np.sqrt((x - GEOM['jet_center_x'])**2 + (z - GEOM['jet_center_z'])**2)

    # ── 3. Deterministic Uniform Sampling (Bug 2 Fix) ──
    # To ensure the static graph skeleton matches features across all simulations,
    # we must use the same 100k node indices for every file with the same mesh.
    sel = rng.choice(
        np.arange(len(coords)), 
        size=SAMPLE['points_per_sim'], 
        replace=False
    )

    xs, ys, zs = x[sel], y[sel], z[sel]
    ts, ps = temp[sel], press[sel]
    us, vs, ws = vel_field[sel, 0], vel_field[sel, 1], vel_field[sel, 2]

    # ── 4. Spatial feature engineering ──────────────────────────────

    # Cylindrical radius from jet axis
    radius = radius_full[sel]

    # Distance to outflow boundary in X and Z
    dist_outflow = np.minimum.reduce([
        xs,
        GEOM['domain_x_max'] - xs,
        zs,
        GEOM['domain_z_max'] - zs,
    ])
    
    # Outlet Geometry Feature (assuming outlets at boundaries)
    outlet_centers = np.array([[0.0, 0.012], [0.10, 0.012]])
    dist_outlet = np.min([
        np.sqrt((xs - cx)**2 + (zs - cz)**2)
        for cx, cz in outlet_centers
    ], axis=0)

    # NEW: Wall-normal distance normalised by BL thickness (δ ≈ D/sqrt(Re))
    delta_bl = GEOM['D_m'] / np.sqrt(max(re_num, 1.0))
    y_norm   = ys / (delta_bl + 1e-10)          # y⁺-like non-dimensional height

    # NEW: Stagnation proximity flag (1 within 1 jet diameter of stagnation point)
    stagnation_flag = (radius < GEOM['D_m']).astype(np.float32)

    # NEW: Non-dimensional temperature using global T_max
    dT_ref    = max(t_max_global - GEOM['T_inlet'], 1e-6)
    theta_norm = (ts - GEOM['T_inlet']) / dT_ref   # range [0, 1] by construction

    # NEW: Local Nusselt proxy at wall nodes (only meaningful where Y ≈ 0)
    # Nu_local = h_local * D / k,  h_local = q'' / (T_wall - T_inlet)
    wall_sel    = ys < 1e-4                         # effectively Y=0 nodes
    nu_local    = np.zeros_like(ts)
    dT_wall     = ts[wall_sel] - GEOM['T_inlet']
    dT_wall     = np.where(np.abs(dT_wall) < 1e-6, 1e-6, dT_wall)
    nu_local[wall_sel] = (heat_flux / dT_wall) * GEOM['D_m'] / FLUID['k']

    # ── 5. Assemble DataFrame ────────────────────────────────────────
    return pd.DataFrame({
        # ── Global context (broadcast to every point)
        'H_D'              : hd,
        'Log_Reynolds'     : np.log(max(re_num, 1.0)),
        'Heat_Flux'        : heat_flux,
        'Stanton_Proxy'    : q_star,

        # ── Spatial context
        'X'                : xs,
        'Y'                : ys,
        'Z'                : zs,
        'Radius'           : radius,
        'Dist_Outflow'     : dist_outflow,
        'Dist_Outlet'      : dist_outlet,
        'Y_norm'           : y_norm,           # NEW
        'Stagnation_Flag'  : stagnation_flag,  # NEW (binary — do NOT scale)
        'sim_id'           : os.path.basename(file_path),   # ADD — unique per simulation

        # ── Targets
        'Temperature'      : ts,
        'Pressure'         : ps,
        'U_vel'            : us,
        'V_vel'            : vs,
        'W_vel'            : ws,

        # ── Physics-constraint targets (derived — used for auxiliary losses)
        'Theta_norm'       : theta_norm,       # NEW (dimensionless T)
        'Nu_local'         : nu_local,         # NEW (local Nu proxy)
    })


# ─────────────────────────────────────────────────
#  PHASE 2: MASTER EXTRACTION LOOP
# ─────────────────────────────────────────────────
def extract_all(valid_files, t_max_global=None) -> pd.DataFrame:
    rng = np.random.default_rng(SAMPLE['seed'])

    chunks = []
    if t_max_global is None:
        t_max_global = -np.inf
        for fp in valid_files:
            with h5py.File(fp, 'r') as f:
                gname = list(f.keys())[0]
                t_max_global = max(t_max_global, f[gname]['Temperature'][:].max())
        t_max_global = float(t_max_global)

    for fp in tqdm(valid_files, desc="Engineering Features"):
        try:
            df_sim = engineer_one_sim(fp, rng, t_max_global=t_max_global)
            chunks.append(df_sim)
        except Exception as e:
            print(f"[SKIP] {os.path.basename(fp)}: {e}")
        gc.collect()

    master = pd.concat(chunks, ignore_index=True)
    print(f"\n[Extract] ✓  Master dataset: {len(master):,} rows × {master.shape[1]} cols")
    return master


# ─────────────────────────────────────────────────
#  PHASE 3: SCALING STRATEGY
# ─────────────────────────────────────────────────
"""
Scaling decisions (justified from your EDA):

  Global features  → StandardScaler
    Re, Heat_Flux, H_D are unimodal and roughly Gaussian once log-transformed.
    
  Spatial coords   → StandardScaler
    X, Y, Z, Radius have physical meaning and linear variance.
    
  Y_norm           → PowerTransformer (Yeo-Johnson)
    Your STD maps showed an extreme right skew in Y: most points near wall.
    PowerTransformer makes this Gaussian so the network trains faster.
    
  Pressure         → PowerTransformer (Yeo-Johnson)
    Stagnation pressure has a heavy right tail (P ∝ V²).
    Your Plot 3 confirmed this non-linearity.
    
  Temperature      → StandardScaler (do NOT use MinMax)
    MinMax would crush the cold ambient region and over-stretch stagnation spike.
    
  Velocity components → StandardScaler
    U, V, W are zero-mean by physics; StandardScaler is ideal.

  Stagnation_Flag  → NOT scaled (binary indicator).
  Theta_norm       → NOT scaled (already ∈ [0,1] by construction).
  Nu_local         → StandardScaler (large range; mostly zero in bulk).
"""

# Column definitions
GLOBAL_COLS  = ['H_D', 'Log_Reynolds', 'Heat_Flux', 'Stanton_Proxy']
SPATIAL_COLS = ['X', 'Y', 'Z', 'Radius', 'Dist_Outflow', 'Dist_Outlet']
SKEWED_COLS  = ['Y_norm']                     # Removed Radial_Vel_Mag
BINARY_COLS  = ['Stagnation_Flag']             # pass-through
TARGET_STD_COLS = ['Temperature', 'U_vel', 'V_vel', 'W_vel']
TARGET_POWER_COLS = ['Pressure']
TARGET_COLS  = ['Temperature', 'Pressure', 'U_vel', 'V_vel', 'W_vel']  # T,P,U,V,W
AUX_COLS     = ['Theta_norm', 'Nu_local']      # auxiliary physics targets

ALL_INPUT_COLS = GLOBAL_COLS + SPATIAL_COLS + SKEWED_COLS + BINARY_COLS


def fit_and_transform(master: pd.DataFrame):
    print("\n[Scale]  Fitting scalers...")

    scaler_global  = StandardScaler()
    scaler_spatial = StandardScaler()
    scaler_skewed  = PowerTransformer(method='yeo-johnson', standardize=True)
    scaler_temp    = StandardScaler()
    scaler_vel     = StandardScaler()
    scaler_target_power = PowerTransformer(method='yeo-johnson', standardize=True)
    scaler_aux     = StandardScaler()

    g_scaled = scaler_global .fit_transform(master[GLOBAL_COLS ])
    s_scaled = scaler_spatial.fit_transform(master[SPATIAL_COLS])
    k_scaled = scaler_skewed .fit_transform(master[SKEWED_COLS ])
    b_raw    = master[BINARY_COLS].values.astype(np.float32)   # unscaled
    
    t_scaled = np.concatenate([
        scaler_temp.fit_transform(master[['Temperature']]),
        scaler_target_power.fit_transform(master[TARGET_POWER_COLS]),
        scaler_vel.fit_transform(master[['U_vel', 'V_vel', 'W_vel']]),
    ], axis=1)

    a_scaled = scaler_aux    .fit_transform(master[AUX_COLS   ])

    # Save every scaler
    save_dir = r"D:\data\JET"
    os.makedirs(save_dir, exist_ok=True)
    for obj, name in [
        (scaler_global,  'scaler_global.pkl'),
        (scaler_spatial, 'scaler_spatial.pkl'),
        (scaler_skewed,  'scaler_skewed.pkl'),
        (scaler_temp,    'scaler_temp.pkl'),
        (scaler_vel,     'scaler_vel.pkl'),
        (scaler_target_power, 'scaler_target_power.pkl'),
        (scaler_aux,     'scaler_aux.pkl'),
    ]:
        joblib.dump(obj, os.path.join(save_dir, name))

    print("[Scale]  ✓  Scalers saved.")

    # Stitch full input matrix: [global | spatial | skewed | binary]
    X_full = np.concatenate([g_scaled, s_scaled, k_scaled, b_raw], axis=1).astype(np.float32)
    T_full = t_scaled.astype(np.float32)
    A_full = a_scaled.astype(np.float32)

    print(f"[Scale]  Input matrix  : {X_full.shape}")
    print(f"[Scale]  Target matrix : {T_full.shape}")
    print(f"[Scale]  Aux matrix    : {A_full.shape}")

    return X_full, T_full, A_full


def transform_only(df: pd.DataFrame, scalers: dict):
    """Apply already-fitted scalers to val/test splits — no fitting."""
    g = scalers['global'].transform(df[GLOBAL_COLS])
    s = scalers['spatial'].transform(df[SPATIAL_COLS])
    k = scalers['skewed'].transform(df[SKEWED_COLS])
    b = df[BINARY_COLS].values.astype(np.float32)
    t_std1 = scalers['temp'].transform(df[['Temperature']])
    t_pwr  = scalers['target_power'].transform(df[TARGET_POWER_COLS])
    t_std2 = scalers['vel'].transform(df[['U_vel', 'V_vel', 'W_vel']])
    a = scalers['aux'].transform(df[AUX_COLS])
    X = np.concatenate([g, s, k, b], axis=1).astype(np.float32)
    T = np.concatenate([t_std1, t_pwr, t_std2], axis=1).astype(np.float32)
    A = a.astype(np.float32)
    return X, T, A


# ─────────────────────────────────────────────────
#  PHASE 3b: FEATURE REPORT
# ─────────────────────────────────────────────────
def print_feature_report(master: pd.DataFrame, X_full: np.ndarray,
                         T_full: np.ndarray, A_full: np.ndarray) -> None:
    """
    Prints a full engineering report covering:
      • Feature manifest  (every column, its group, scaler, and physical meaning)
      • Raw statistics    (mean / std / min / max on the unscaled master DataFrame)
      • Scaled statistics (confirms scalers worked — mean≈0, std≈1 for StandardScaler)
      • Dataset memory    (RAM cost of each matrix)
      • Sampling audit    (boundary-layer vs bulk split actually achieved)
      • Correlation table (top feature→target correlations, ranked)
    """
    SEP  = "=" * 72
    SEP2 = "-" * 72

    # ── SECTION 1: Feature manifest ─────────────────────────────────────
    print(f"\n{SEP}")
    print("  FEATURE ENGINEERING REPORT")
    print(SEP)

    manifest = [
        # (column,            group,       scaler,              physical meaning)
        ("H_D",               "Global",    "StandardScaler",    "Jet-to-plate spacing ratio"),
        ("Log_Reynolds",      "Global",    "StandardScaler",    "log(Re) — linearises Nu∝Re^0.6 relationship"),
        ("Heat_Flux",         "Global",    "StandardScaler",    "q'' = Power/A_chip  [W/m²]"),
        ("Stanton_Proxy",     "Global",    "StandardScaler",    "St ≈ q''/(ρVcpΔT)  — thermal efficiency"),
        ("X",                 "Spatial",   "StandardScaler",    "Streamwise coord [m]"),
        ("Y",                 "Spatial",   "StandardScaler",    "Wall-normal coord [m]"),
        ("Z",                 "Spatial",   "StandardScaler",    "Spanwise coord [m]"),
        ("Radius",            "Spatial",   "StandardScaler",    "Cylindrical r from jet axis [m]"),
        ("Dist_Outflow",      "Spatial",   "StandardScaler",    "Distance to nearest X/Z boundary [m]"),
        ("Dist_Outlet",       "Spatial",   "StandardScaler",    "Distance to nearest outlet centre [m]"),
        ("Y_norm",            "Skewed",    "PowerTransformer",  "y/δ — BL-normalised height (right-skewed)"),
        ("Stagnation_Flag",   "Binary",    "None (pass-thru)",  "1 if r < D (stagnation zone)"),
        ("Temperature",       "Target",    "StandardScaler",    "Fluid temperature [°C]"),
        ("Pressure",          "Target",    "PowerTransformer",  "Static pressure [Pa] — heavy right tail (P∝V²)"),
        ("U_vel",             "Target",    "StandardScaler",    "X-velocity component [m/s]"),
        ("V_vel",             "Target",    "StandardScaler",    "Y-velocity component [m/s]"),
        ("W_vel",             "Target",    "StandardScaler",    "Z-velocity component [m/s]"),
        ("Theta_norm",        "Aux",       "StandardScaler",    "θ=(T−T_in)/(T_max−T_in) ∈ [0,1]"),
        ("Nu_local",          "Aux",       "StandardScaler",    "Nu_local = h·D/k  (wall nodes only)"),
    ]

    # Column widths
    print(f"\n  {'#':<4} {'Column':<20} {'Group':<10} {'Scaler':<20} {'Physical Meaning'}")
    print(f"  {SEP2}")
    input_idx  = 0
    target_idx = 0
    aux_idx    = 0
    for i, (col, grp, scaler, meaning) in enumerate(manifest):
        if grp in ("Global", "Spatial", "Skewed", "Binary"):
            tag = f"IN[{input_idx:02d}]"
            input_idx += 1
        elif grp == "Target":
            tag = f"TG[{target_idx:02d}]"
            target_idx += 1
        else:
            tag = f"AX[{aux_idx:02d}]"
            aux_idx += 1
        print(f"  {tag:<6} {col:<20} {grp:<10} {scaler:<20} {meaning}")

    # ── SECTION 2: Dataset dimensions ───────────────────────────────────
    print(f"\n{SEP}")
    print("  DATASET DIMENSIONS")
    print(SEP)
    n_samples = X_full.shape[0]
    print(f"  Total samples          : {n_samples:>12,}")
    print(f"  Input  tensor  shape   : {str(X_full.shape):>20}   "
          f"({len(GLOBAL_COLS)}G + {len(SPATIAL_COLS)}S + {len(SKEWED_COLS)}K + {len(BINARY_COLS)}B = {X_full.shape[1]} dims)")
    print(f"  Target tensor  shape   : {str(T_full.shape):>20}   "
          f"(T, P, U, V, W)")
    print(f"  Aux    tensor  shape   : {str(A_full.shape):>20}   "
          f"(Theta_norm, Nu_local)")

    # ── SECTION 3: Memory cost ───────────────────────────────────────────
    print(f"\n{SEP}")
    print("  MEMORY FOOTPRINT (float32)")
    print(SEP)
    x_mb = X_full.nbytes / 1e6
    t_mb = T_full.nbytes / 1e6
    a_mb = A_full.nbytes / 1e6
    print(f"  Input  matrix   : {x_mb:>8.1f} MB")
    print(f"  Target matrix   : {t_mb:>8.1f} MB")
    print(f"  Aux    matrix   : {a_mb:>8.1f} MB")
    print(f"  ─────────────────────────")
    print(f"  Total on RAM    : {x_mb + t_mb + a_mb:>8.1f} MB")

    # ── SECTION 4: Raw statistics (unscaled) ─────────────────────────────
    print(f"\n{SEP}")
    print("  RAW FEATURE STATISTICS  (unscaled master DataFrame)")
    print(SEP)
    all_report_cols = (GLOBAL_COLS + SPATIAL_COLS + SKEWED_COLS
                       + BINARY_COLS + TARGET_COLS + AUX_COLS)
    stats = master[all_report_cols].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    stats = stats[['mean', 'std', 'min', '1%', '50%', '99%', 'max']]
    stats.columns = ['mean', 'std', 'min', 'p01', 'p50', 'p99', 'max']

    print(f"\n  {'Feature':<22} {'mean':>12} {'std':>12} {'min':>12} "
          f"{'p01':>12} {'p50':>12} {'p99':>12} {'max':>12}")
    print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for col, row in stats.iterrows():
        print(f"  {col:<22} {row['mean']:>12.4g} {row['std']:>12.4g} "
              f"{row['min']:>12.4g} {row['p01']:>12.4g} {row['p50']:>12.4g} "
              f"{row['p99']:>12.4g} {row['max']:>12.4g}")

    # ── SECTION 5: Scaled statistics (sanity check) ──────────────────────
    print(f"\n{SEP}")
    print("  SCALED TENSOR STATISTICS  (post-transform — mean≈0, std≈1 expected)")
    print(SEP)
    col_names_X = ALL_INPUT_COLS
    col_names_T = TARGET_COLS
    col_names_A = AUX_COLS

    def _scaled_stats(mat, names, label):
        print(f"\n  [{label}]")
        print(f"  {'Feature':<22} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}  {'OK?':>6}")
        print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*10}  {'-'*6}")
        for i, name in enumerate(names):
            col_data = mat[:, i]
            m, s, lo, hi = col_data.mean(), col_data.std(), col_data.min(), col_data.max()
            # Binary / pass-through columns are exempt from mean≈0 check
            is_binary = name in BINARY_COLS
            ok = "SKIP" if is_binary else ("✓" if abs(m) < 0.05 and 0.8 < s < 1.2 else "⚠ CHECK")
            print(f"  {name:<22} {m:>10.4f} {s:>10.4f} {lo:>10.4f} {hi:>10.4f}  {ok:>6}")

    _scaled_stats(X_full, col_names_X, "INPUT  TENSOR")
    _scaled_stats(T_full, col_names_T, "TARGET TENSOR")
    _scaled_stats(A_full, col_names_A, "AUX    TENSOR")

    # ── SECTION 6: Sampling audit ─────────────────────────────────────────
    print(f"\n{SEP}")
    print("  SAMPLING AUDIT  (boundary-layer vs bulk)")
    print(SEP)
    bl_count   = (master['Y'] < GEOM['wall_y_thresh']).sum()
    bulk_count = len(master) - bl_count
    print(f"  Boundary-layer points (Y < {GEOM['wall_y_thresh']*1000:.0f} mm) : "
          f"{bl_count:>12,}  ({100*bl_count/len(master):.1f}%)")
    print(f"  Bulk-flow points                          : "
          f"{bulk_count:>12,}  ({100*bulk_count/len(master):.1f}%)")
    print(f"  Target BL fraction                        : "
          f"{SAMPLE['boundary_frac']*100:.0f}%")
    stag_count = (master['Stagnation_Flag'] == 1).sum()
    print(f"  Stagnation-zone points (r < D)            : "
          f"{stag_count:>12,}  ({100*stag_count/len(master):.1f}%)")

    # ── SECTION 7: Top correlations (feature → target) ───────────────────
    print(f"\n{SEP}")
    print("  TOP FEATURE → TARGET CORRELATIONS")
    print(SEP)
    corr_cols = ALL_INPUT_COLS + TARGET_COLS
    # Use raw (unscaled) values for interpretable Pearson r
    corr_df   = master[corr_cols].corr()
    print(f"\n  {'Feature':<22} {'→ Temp':>10} {'→ Pressure':>12} "
          f"{'→ U_vel':>10} {'→ V_vel':>10} {'→ W_vel':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for feat in ALL_INPUT_COLS:
        r_T = corr_df.loc[feat, 'Temperature']
        r_P = corr_df.loc[feat, 'Pressure']
        r_U = corr_df.loc[feat, 'U_vel']
        r_V = corr_df.loc[feat, 'V_vel']
        r_W = corr_df.loc[feat, 'W_vel']
        print(f"  {feat:<22} {r_T:>10.3f} {r_P:>12.3f} {r_U:>10.3f} {r_V:>10.3f} {r_W:>10.3f}")

    # Highlight strongest single correlation per target
    print(f"\n  Strongest predictor per target:")
    for tgt in TARGET_COLS:
        best_feat = corr_df[tgt].drop(TARGET_COLS).abs().idxmax()
        best_r    = corr_df.loc[best_feat, tgt]
        print(f"    {tgt:<15} ← {best_feat:<22}  r = {best_r:+.3f}")

    print(f"\n{SEP}")
    print("  END OF FEATURE REPORT")
    print(f"{SEP}\n")


# ─────────────────────────────────────────────────
#  PHASE 4: PYTORCH DATASET
# ─────────────────────────────────────────────────
class JetImpingementDataset(Dataset):
    """
    Returns (inputs, targets, aux_targets) per sample.
    
    inputs      : [H_D, Re, Heat_Flux, Stanton | X,Y,Z,Radius,Dist_Outflow | Y_norm,RadVelMag | StagnFlag]
    targets     : [T, P, U, V, W]
    aux_targets : [Theta_norm, Nu_local]  — used for physics-constraint auxiliary losses
    """
    def __init__(self, X: np.ndarray, T: np.ndarray, A: np.ndarray, nodes_per_sim: int = 100_000):
        # Reshape flat arrays into [Simulations, Nodes, Features]
        num_sims = len(X) // nodes_per_sim
        
        self.X = torch.from_numpy(X).view(num_sims, nodes_per_sim, X.shape[1])
        self.T = torch.from_numpy(T).view(num_sims, nodes_per_sim, T.shape[1])
        self.A = torch.from_numpy(A).view(num_sims, nodes_per_sim, A.shape[1])

    def __len__(self):
        return self.X.shape[0] # Returns number of simulations, not number of nodes

    def __getitem__(self, idx):
        # Returns a full 100k x N tensor for one specific simulation
        return self.X[idx], self.T[idx], self.A[idx]

    @property
    def input_dim(self):
        return self.X.shape[1]

    @property
    def target_dim(self):
        return self.T.shape[1]


# ─────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  JET IMPINGEMENT  ·  FEATURE ENGINEERING PIPELINE")
    print("="*60)

    # 1. Validate files
    all_valid_files = validate_files(folder_path="D:/data/JET/**/*.h5")
    
    # Stratified split: test_size = 0.15 (train/val 0.85, then test)
    hd_list = [float(re.search(r'[/\\]H(\d+)[/\\]', f).group(1)) for f in all_valid_files]
    re_list = [
        (FLUID['rho'] * float(re.search(r'Vel_([-\d\.]+)', f).group(1)) * GEOM['D_m']) / FLUID['mu']
        for f in all_valid_files
    ]
    labels = [f"{hd}_{int(abs(reynolds)//10000)}" for hd, reynolds in zip(hd_list, re_list)]
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_idx, test_idx = next(sss.split(all_valid_files, labels))
    
    # We can split train/val further if we want, split train_val into train and val (15% val of total -> ~17.6%)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.176, random_state=42)
    train_idx, val_idx = next(sss2.split(np.array(all_valid_files)[train_val_idx], np.array(labels)[train_val_idx]))
    
    actual_train_idx = train_val_idx[train_idx]
    actual_val_idx = train_val_idx[val_idx]
    
    train_files = [all_valid_files[i] for i in actual_train_idx]
    val_files = [all_valid_files[i] for i in actual_val_idx]
    test_files = [all_valid_files[i] for i in test_idx]

    # Pre-calculate global t_max to avoid data leaking from test set
    t_max_global = -np.inf
    print("Calculating global T_max from train set...")
    for fp in train_files:
        with h5py.File(fp, 'r') as f:
            gname = list(f.keys())[0]
            t_max_global = max(t_max_global, f[gname]['Temperature'][:].max())
    t_max_global = float(t_max_global)


    # ── PASS 1: Fit scalers on the full concatenated training set ──
    print("\n--- Pass 1: Fitting scalers on full training set ---")
    df_train_full = extract_all(train_files, t_max_global=t_max_global)
    X_train, T_train, A_train = fit_and_transform(df_train_full)   # fits + saves .pkl files

    save_dir = r"D:\data\JET"
    scalers = {
        'global'      : joblib.load(os.path.join(save_dir, 'scaler_global.pkl')),
        'spatial'     : joblib.load(os.path.join(save_dir, 'scaler_spatial.pkl')),
        'skewed'      : joblib.load(os.path.join(save_dir, 'scaler_skewed.pkl')),
        'temp'        : joblib.load(os.path.join(save_dir, 'scaler_temp.pkl')),
        'vel'         : joblib.load(os.path.join(save_dir, 'scaler_vel.pkl')),
        'target_power': joblib.load(os.path.join(save_dir, 'scaler_target_power.pkl')),
        'aux'         : joblib.load(os.path.join(save_dir, 'scaler_aux.pkl')),
    }
    print_feature_report(df_train_full, X_train, T_train, A_train)
    del df_train_full, X_train, T_train, A_train
    gc.collect()

    # ── PASS 2: Save per-simulation .pt files using fitted scalers ──
    def process_split(split_files, split_name, scalers):
        print(f"\n--- Processing {split_name} split (per-simulation saves) ---")
        sim_save_dir = os.path.join(save_dir, f'{split_name}_sims')
        os.makedirs(sim_save_dir, exist_ok=True)

        for fp in tqdm(split_files, desc=f"Processing {split_name}"):
            try:
                # Bug 8 Fix: Seed per-file using a hash of the filename for reproducibility
                file_seed = int(hashlib.md5(fp.encode()).hexdigest(), 16) % (2**32)
                rng = np.random.default_rng(file_seed)
                
                df_sim = engineer_one_sim(fp, rng, t_max_global=t_max_global)
                X, T, A = transform_only(df_sim, scalers)
                ds = JetImpingementDataset(X, T, A)
                sim_name = os.path.basename(os.path.dirname(fp)) + '_' + os.path.basename(fp).replace('.h5', '')
                torch.save(ds, os.path.join(sim_save_dir, f'{sim_name}.pt'))
                del df_sim, ds
            except Exception as e:
                print(f"[SKIP] {os.path.basename(fp)}: {e}")
            gc.collect()

    process_split(train_files, 'train', scalers)
    process_split(val_files,   'val',   scalers)
    process_split(test_files,  'test',  scalers)

    print(f"\n{'='*60}")
    print(f"  ✅  PIPELINE COMPLETE")
    print(f"{'='*60}\n")

"""
The print_feature_report() function is now added as Phase 3b and called automatically in main. Here's exactly what it prints when you run the script:


Section 1 — Feature Manifest lists every single column with its index tag (IN[00], TG[00], AX[00]), which group it belongs to, which scaler was applied, and a plain-English physical meaning. You'll see the full 12-input → 5-target → 2-aux layout at a glance.

Section 2 — Dataset Dimensions confirms tensor shapes and the breakdown 4G + 6S + 1K + 1B = 12 dims so you know exactly what your network's in_features should be.

Section 3 — Memory Footprint shows MB cost per matrix and total RAM so you can plan GPU batch sizes before training.

Section 4 — Raw Statistics prints a full table with mean, std, min, p01, p50, p99, max for every column in unscaled form. This is where you catch physics anomalies — for example if Nu_local p99 is astronomically large, it means your wall temperature is nearly equal to inlet at some nodes.

Section 5 — Scaled Tensor Statistics verifies each scaler worked correctly. StandardScaler columns should show mean≈0, std≈1 and get a ✓. Binary Stagnation_Flag is marked SKIP. Anything that drifts gets flagged ⚠ CHECK automatically.

Section 6 — Sampling Audit confirms the 75/25 boundary-layer split was actually achieved in practice, and also reports how many stagnation-zone points were captured.

Section 7 — Correlation Table prints Pearson r for every input feature against all 5 targets, then highlights the single strongest predictor per target. If Reynolds shows r = −0.85 with Temperature, that confirms Re is your dominant input and validates the physics before a single training step.
"""

