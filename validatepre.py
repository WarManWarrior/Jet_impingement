"""
=============================================================================
  VALIDATION SCRIPT — H5 Ground Truth vs Model Predictions
  Compares jet_prediction_results.csv against actual CFD HDF5 data
=============================================================================

Usage:
    python validate_predictions.py \
        --h5     path/to/actual_cfd_data.h5 \
        --pred   jet_prediction_results.csv \
        --out    results/validation

The script:
  1. Inspects the H5 file structure automatically
  2. Extracts T, P, U/V/W fields from the H5
  3. Spatially aligns predicted points to nearest H5 nodes (KD-tree)
  4. Computes per-field error metrics (MAE, RMSE, R², Max Error, MAPE)
  5. Generates comparison scatter plots + error distribution plots
  6. Saves a summary CSV and a full point-wise error CSV
=============================================================================
"""

import os
import argparse
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.metrics import r2_score

# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Validate surrogate predictions vs H5 CFD data.")
    p.add_argument("--h5",   required=True,  help="Path to the ground-truth HDF5 file")
    p.add_argument("--pred", required=True,  help="Path to jet_prediction_results.csv")
    p.add_argument("--out",  default="results/validation", help="Output directory")
    p.add_argument("--k",    type=int, default=1,
                   help="Number of nearest H5 neighbours to average (default 1 = exact nearest)")
    p.add_argument("--max_dist", type=float, default=None,
                   help="Max allowable spatial distance for a valid match (metres). "
                        "Points further than this are flagged as unmatched.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# H5 structure inspection
# ─────────────────────────────────────────────────────────────────────────────

def inspect_h5(path: str) -> dict:
    """
    Walk the H5 file and return a dict of all dataset paths with their shapes.
    Prints a human-readable tree so you can see exactly what keys are available.
    """
    print(f"\n{'='*60}")
    print(f"  H5 STRUCTURE: {os.path.basename(path)}")
    print(f"{'='*60}")
    datasets = {}

    def _visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets[name] = obj.shape
            print(f"  /{name:50s}  shape={obj.shape}  dtype={obj.dtype}")

    with h5py.File(path, "r") as f:
        f.visititems(_visitor)

    print(f"{'='*60}\n")
    return datasets


# ─────────────────────────────────────────────────────────────────────────────
# H5 field extraction
# ─────────────────────────────────────────────────────────────────────────────

# Candidate key names for each physical field.
# The script tries each in order and uses the first one found.
CANDIDATE_KEYS = {
    "X":    ["x", "X", "coords/x", "Coordinates/x", "Points/x", "mesh/x", "geometry/x"],
    "Y":    ["y", "Y", "coords/y", "Coordinates/y", "Points/y", "mesh/y", "geometry/y"],
    "Z":    ["z", "Z", "coords/z", "Coordinates/z", "Points/z", "mesh/z", "geometry/z"],
    "T":    ["Temperature", "temperature", "T", "temp", "T_C", "temperature_C",
             "fields/T", "data/Temperature"],
    "P":    ["Pressure", "pressure", "P", "p", "fields/P", "data/Pressure"],
    "U":    ["U", "u", "Vx", "velocity_x", "fields/U", "data/U"],
    "V":    ["V", "v", "Vy", "velocity_y", "fields/V", "data/V"],
    "W":    ["W", "w", "Vz", "velocity_z", "fields/W", "data/W"],
    "Vmag": ["Velocity_Magnitude", "velocity_magnitude", "speed",
             "fields/Vmag", "data/VelocityMagnitude"],
}


def _find_key(h5file, candidates: list):
    """Return the first candidate key that exists in the H5 file, else None."""
    for k in candidates:
        if k in h5file:
            return k
    return None


def _get_group(h5file) -> h5py.Group:
    """
    Return the data group to search inside.
    If the file has a single top-level group (e.g. 'Vel_-4m_per_sec_Pow_70W'),
    descend into it automatically so all dataset paths work relative to that group.
    """
    keys = list(h5file.keys())
    if len(keys) == 1 and isinstance(h5file[keys[0]], h5py.Group):
        print(f"  Auto-descending into top-level group: '{keys[0]}'")
        return h5file[keys[0]]
    return h5file


def load_h5_fields(path: str, datasets: dict) -> pd.DataFrame:
    """
    Load coordinate and field arrays from the H5 file.
    Handles:
      - Grouped H5 files (single top-level group containing all datasets)
      - Combined Coordinates array of shape (N, 3) → split into X, Y, Z
      - Combined Velocity array of shape (N, 3)  → split into U, V, W
      - Scalar field arrays of shape (N,) or (N, 1)
    Returns a DataFrame with columns X, Y, Z, T_actual, P_actual,
    U_actual, V_actual, W_actual, Vmag_actual.
    """
    print("Loading ground-truth fields from H5 ...")
    data = {}

    with h5py.File(path, "r") as f:
        grp = _get_group(f)

        # ── Coordinates ───────────────────────────────────────────────────────
        # Try combined (N,3) array first, then individual axis keys
        coord_candidates = ["Coordinates", "coordinates", "Points", "coords",
                            "mesh/Coordinates", "geometry/Coordinates"]
        coord_key = _find_key(grp, coord_candidates)

        if coord_key is not None:
            coords = grp[coord_key][:]          # (N, 3)
            assert coords.ndim == 2 and coords.shape[1] >= 3, \
                f"Expected (N,3) coordinates, got shape {coords.shape}"
            data["X"] = coords[:, 0].astype(np.float64)
            data["Y"] = coords[:, 1].astype(np.float64)
            data["Z"] = coords[:, 2].astype(np.float64)
            print(f"  Loaded '{coord_key}' → X, Y, Z  shape={coords.shape}")
        else:
            # Fall back to separate per-axis keys
            for i, col in enumerate(["X", "Y", "Z"]):
                key = _find_key(grp, CANDIDATE_KEYS[col])
                if key is None:
                    raise KeyError(
                        f"Cannot find {col} coordinates in H5.\n"
                        f"Tried combined keys: {coord_candidates}\n"
                        f"Tried per-axis keys: {CANDIDATE_KEYS[col]}\n"
                        f"Available keys in group: {list(grp.keys())}\n"
                        "Add the correct key to CANDIDATE_KEYS or coord_candidates."
                    )
                data[col] = grp[key][:].ravel().astype(np.float64)

        n_points = len(data["X"])
        print(f"  Found {n_points:,} spatial points in H5")

        # ── Velocity (combined N×3 or separate components) ────────────────────
        vel_candidates = ["Velocity", "velocity", "UVW", "fields/Velocity"]
        vel_key = _find_key(grp, vel_candidates)

        if vel_key is not None:
            vel = grp[vel_key][:]               # (N, 3)
            if vel.ndim == 2 and vel.shape[1] == 3:
                data["U_actual"] = vel[:, 0].astype(np.float64)
                data["V_actual"] = vel[:, 1].astype(np.float64)
                data["W_actual"] = vel[:, 2].astype(np.float64)
                print(f"  Loaded '{vel_key}' → U, V, W  shape={vel.shape}  "
                      f"mag range [{np.sqrt((vel**2).sum(axis=1)).min():.3g}, "
                      f"{np.sqrt((vel**2).sum(axis=1)).max():.3g}] m/s")
            else:
                # Treat as magnitude
                data["U_actual"] = vel.ravel().astype(np.float64)
                data["V_actual"] = np.zeros(n_points)
                data["W_actual"] = np.zeros(n_points)
        else:
            # Try separate component keys
            for comp, field in [("U_actual","U"), ("V_actual","V"), ("W_actual","W")]:
                key = _find_key(grp, CANDIDATE_KEYS[field])
                if key is not None:
                    data[comp] = grp[key][:].ravel()[:n_points].astype(np.float64)
                else:
                    print(f"  WARNING: {field} not found — setting {comp} to NaN")
                    data[comp] = np.full(n_points, np.nan)

        # ── Scalar fields: Temperature, Pressure ──────────────────────────────
        scalar_map = {
            "T_actual": ["Temperature", "temperature", "T", "temp",
                         "fields/Temperature", "data/Temperature"],
            "P_actual": ["Pressure", "pressure", "P", "p",
                         "fields/Pressure", "data/Pressure"],
        }
        for col, candidates in scalar_map.items():
            key = _find_key(grp, candidates)
            if key is not None:
                arr = grp[key][:].ravel()[:n_points].astype(np.float64)
                data[col] = arr
                print(f"  Loaded '{key}' → {col}  [{arr.min():.3g}, {arr.max():.3g}]")
            else:
                print(f"  WARNING: no match for {col} — setting to NaN")
                data[col] = np.full(n_points, np.nan)

    df = pd.DataFrame(data)

    # Derive Vmag
    df["Vmag_actual"] = np.sqrt(
        df["U_actual"]**2 + df["V_actual"]**2 + df["W_actual"]**2
    )
    print(f"  Derived Vmag_actual  [{df['Vmag_actual'].min():.3g}, "
          f"{df['Vmag_actual'].max():.3g}] m/s")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Spatial alignment via KD-tree
# ─────────────────────────────────────────────────────────────────────────────

def align_predictions(df_pred: pd.DataFrame, df_h5: pd.DataFrame,
                      k: int = 1, max_dist: float = None) -> pd.DataFrame:
    """
    For each predicted point, find the k nearest H5 nodes and average their
    field values. This handles cases where the meshes don't share the same nodes.

    Returns a merged DataFrame with both predicted and matched actual values,
    plus a 'match_dist' column showing the nearest-neighbour distance.
    """
    print(f"\nSpatially aligning predictions to H5 mesh (k={k} neighbours) ...")

    h5_coords  = df_h5[["X", "Y", "Z"]].values
    pred_coords = df_pred[["X", "Y", "Z"]].values

    tree = cKDTree(h5_coords)
    dists, idxs = tree.query(pred_coords, k=k, workers=-1)  # parallel

    if k == 1:
        dists = dists[:, np.newaxis]
        idxs  = idxs[:, np.newaxis]

    # Average over k neighbours (weighted by inverse distance if k > 1)
    if k == 1:
        weights = np.ones_like(dists)
    else:
        weights = 1.0 / (dists + 1e-12)
        weights /= weights.sum(axis=1, keepdims=True)

    actual_cols = ["T_actual", "P_actual", "U_actual", "V_actual",
                   "W_actual", "Vmag_actual"]
    matched = {}
    for col in actual_cols:
        vals = df_h5[col].values[idxs]          # (N_pred, k)
        matched[col] = (vals * weights).sum(axis=1)

    matched["match_dist"] = dists[:, 0]          # nearest-neighbour distance

    df_matched = pd.DataFrame(matched, index=df_pred.index)
    df_out = pd.concat([df_pred.reset_index(drop=True),
                        df_matched.reset_index(drop=True)], axis=1)

    if max_dist is not None:
        n_bad = (df_out["match_dist"] > max_dist).sum()
        print(f"  Points beyond max_dist ({max_dist:.4g} m): {n_bad:,} "
              f"({100*n_bad/len(df_out):.1f}%) — flagged as NaN")
        mask = df_out["match_dist"] > max_dist
        for col in actual_cols:
            df_out.loc[mask, col] = np.nan

    print(f"  Alignment complete. Median match distance: "
          f"{df_out['match_dist'].median():.4g} m")
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# Error metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    """Compute MAE, RMSE, R², Max absolute error, MAPE for one field."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]

    if len(yt) == 0:
        print(f"  {label}: no valid points — skipping")
        return {}

    mae   = np.mean(np.abs(yt - yp))
    rmse  = np.sqrt(np.mean((yt - yp)**2))
    r2    = r2_score(yt, yp)
    maxe  = np.max(np.abs(yt - yp))
    # MAPE — guard against zero-valued ground truth
    nonzero = np.abs(yt) > 1e-8
    mape = np.mean(np.abs((yt[nonzero] - yp[nonzero]) / yt[nonzero])) * 100 if nonzero.sum() > 0 else np.nan

    print(f"  {label:30s}  MAE={mae:.4g}  RMSE={rmse:.4g}  "
          f"R²={r2:.4f}  MaxErr={maxe:.4g}  MAPE={mape:.2f}%")

    return {
        "Field":  label,
        "N":      int(mask.sum()),
        "MAE":    mae,
        "RMSE":   rmse,
        "R2":     r2,
        "MaxErr": maxe,
        "MAPE_%": mape,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

FIELD_PAIRS = [
    # (predicted_col,          actual_col,     label,                unit)
    ("Temperature_C",          "T_actual",     "Temperature",        "°C"),
    ("Pressure_Pa",            "P_actual",     "Pressure",           "Pa"),
    ("U_vel",                  "U_actual",     "Velocity U",         "m/s"),
    ("V_vel",                  "V_actual",     "Velocity V",         "m/s"),
    ("W_vel",                  "W_actual",     "Velocity W",         "m/s"),
    ("Velocity_Magnitude",     "Vmag_actual",  "Velocity Magnitude", "m/s"),
]


def make_scatter_plot(df: pd.DataFrame, pred_col: str, actual_col: str,
                      label: str, unit: str, out_path: str):
    """Predicted vs actual scatter with identity line and R² annotation."""
    mask = ~(df[pred_col].isna() | df[actual_col].isna())
    yt = df.loc[mask, actual_col].values
    yp = df.loc[mask, pred_col].values
    if len(yt) == 0:
        return

    r2 = r2_score(yt, yp)
    lim = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yt, yp, s=2, alpha=0.3, rasterized=True, color="#185FA5")
    ax.plot(lim, lim, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel(f"CFD (ground truth) [{unit}]")
    ax.set_ylabel(f"Model prediction [{unit}]")
    ax.set_title(f"{label}  —  R² = {r2:.4f}")
    ax.legend(fontsize=9)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def make_error_histogram(df: pd.DataFrame, pred_col: str, actual_col: str,
                         label: str, unit: str, out_path: str):
    """Histogram of signed errors (predicted − actual)."""
    mask = ~(df[pred_col].isna() | df[actual_col].isna())
    err = (df.loc[mask, pred_col] - df.loc[mask, actual_col]).values
    if len(err) == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(err, bins=80, color="#1D9E75", edgecolor="none", alpha=0.85)
    ax.axvline(0,         color="red",    linewidth=1.5, linestyle="--", label="Zero error")
    ax.axvline(err.mean(), color="#D85A30", linewidth=1.5, linestyle="--",
               label=f"Mean = {err.mean():.3g}")
    ax.set_xlabel(f"Error (predicted − actual) [{unit}]")
    ax.set_ylabel("Count")
    ax.set_title(f"{label} error distribution  σ={err.std():.3g}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def make_spatial_error_plot(df: pd.DataFrame, pred_col: str, actual_col: str,
                            label: str, unit: str, out_path: str):
    """
    2-D top-down view (X–Z plane) coloured by absolute error magnitude.
    Useful for spotting spatial bias (e.g. near the jet stagnation zone).
    """
    mask = ~(df[pred_col].isna() | df[actual_col].isna())
    sub  = df[mask].copy()
    sub["abs_err"] = np.abs(sub[pred_col] - sub[actual_col])

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(sub["X"], sub["Z"], c=sub["abs_err"],
                    s=2, cmap="hot_r", rasterized=True,
                    vmin=0, vmax=sub["abs_err"].quantile(0.98))
    plt.colorbar(sc, ax=ax, label=f"|Error| [{unit}]")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(f"{label} — Spatial absolute error (top-down view, Y averaged)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 1. Inspect H5 structure
    datasets = inspect_h5(args.h5)

    # 2. Load H5 ground truth
    df_h5 = load_h5_fields(args.h5, datasets)

    # 3. Load model predictions
    print(f"Loading predictions from {args.pred} ...")
    df_pred = pd.read_csv(args.pred)
    print(f"  Prediction points: {len(df_pred):,}")
    print(f"  Columns: {list(df_pred.columns)}")

    # Validate required coordinate columns
    for col in ["X", "Y", "Z"]:
        if col not in df_pred.columns:
            raise KeyError(f"Prediction CSV missing column '{col}'. "
                           f"Found: {list(df_pred.columns)}")

    # 4. Spatial alignment
    df = align_predictions(df_pred, df_h5, k=args.k, max_dist=args.max_dist)

    # 5. Metrics
    print(f"\n{'='*60}")
    print("  ERROR METRICS")
    print(f"{'='*60}")
    summary_rows = []
    for pred_col, actual_col, label, unit in FIELD_PAIRS:
        if pred_col not in df.columns or actual_col not in df.columns:
            print(f"  Skipping {label} (column missing)")
            continue
        row = compute_metrics(df[actual_col].values, df[pred_col].values, label)
        if row:
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.out, "validation_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"\nSummary saved → {summary_path}")
    print(summary_df.to_string(index=False))

    # 6. Per-point error CSV
    for pred_col, actual_col, label, _ in FIELD_PAIRS:
        if pred_col in df.columns and actual_col in df.columns:
            df[f"err_{label.replace(' ','_')}"] = df[pred_col] - df[actual_col]
            df[f"abs_err_{label.replace(' ','_')}"] = np.abs(df[pred_col] - df[actual_col])

    pointwise_path = os.path.join(args.out, "pointwise_errors.csv")
    df.to_csv(pointwise_path, index=False, encoding="utf-8")
    print(f"Point-wise errors saved → {pointwise_path}")

    # 7. Plots
    print("\nGenerating plots ...")
    for pred_col, actual_col, label, unit in FIELD_PAIRS:
        if pred_col not in df.columns or actual_col not in df.columns:
            continue
        slug = label.replace(" ", "_")
        make_scatter_plot(df, pred_col, actual_col, label, unit,
                          os.path.join(args.out, f"scatter_{slug}.png"))
        make_error_histogram(df, pred_col, actual_col, label, unit,
                             os.path.join(args.out, f"hist_{slug}.png"))
        make_spatial_error_plot(df, pred_col, actual_col, label, unit,
                                os.path.join(args.out, f"spatial_err_{slug}.png"))
        print(f"  Plots saved for {label}")

    # 8. Match distance distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["match_dist"].values, bins=60, color="#534AB7", edgecolor="none", alpha=0.85)
    ax.set_xlabel("Nearest-neighbour distance to H5 mesh node [m]")
    ax.set_ylabel("Count")
    ax.set_title("Spatial alignment quality — match distance distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "match_distances.png"), dpi=150)
    plt.close()

    print(f"\n{'='*60}")
    print(f"  Validation complete. All outputs in: {args.out}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()