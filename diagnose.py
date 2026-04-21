"""
DIAGNOSTIC SCRIPT — Jet Impingement Surrogate (16-feature v2)
Compatible with your current FeatureEngineering.py + LatentGNN.py + train.py
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from pathlib import Path
import re

# ─────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────
DATA_DIR = r"D:\data\JET"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pick any validation file (change index if you want another sim)
val_dir = Path(DATA_DIR) / "val_sims"
val_files = sorted(val_dir.glob("*.pt"))
val_path = val_files[0]      # ← change [0] here if needed
print(f"Diagnosing validation file:\n{val_path.name}\n")

# ─────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────
from FeatureEngineering import (
    inverse_log_transform_pressure,
    GEOM,
    JetImpingementDataset   # required for loading .pt files
)
import joblib
from LatentGNN import JetLatentGNN   # your current v2 model

# ─────────────────────────────────────────────────
#  ROBUST H/D DETECTION
# ─────────────────────────────────────────────────
hd_match = re.search(r'H(\d+)', str(val_path))
if hd_match:
    hd_val = int(hd_match.group(1))
    print(f"Detected H/D = {hd_val}")
else:
    hd_val = 6
    print(f"⚠ No H/D found → defaulting to H/D = {hd_val}")

# ─────────────────────────────────────────────────
#  LOAD EVERYTHING
# ─────────────────────────────────────────────────
print("\nLoading scalers, model, and graph template...")

scalers = {
    'global': joblib.load(os.path.join(DATA_DIR, 'scaler_global.pkl')),
    'spatial': joblib.load(os.path.join(DATA_DIR, 'scaler_spatial.pkl')),
    'skewed': joblib.load(os.path.join(DATA_DIR, 'scaler_skewed.pkl')),
    'temp': joblib.load(os.path.join(DATA_DIR, 'scaler_temp.pkl')),
    'vel': joblib.load(os.path.join(DATA_DIR, 'scaler_vel.pkl')),
    'pressure': joblib.load(os.path.join(DATA_DIR, 'scaler_pressure.pkl')),
}

t_max_global = float(np.load(os.path.join(DATA_DIR, 't_max_global.npy'))[0])

# v2 model — 16 features
model = JetLatentGNN(in_features=16, hidden_features=128, out_features=5).to(DEVICE)
checkpoint = torch.load(
    os.path.join(DATA_DIR, "best_jet_surrogate_model.pth"),
    weights_only=False,
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"  Loaded model from epoch {checkpoint.get('epoch', '?')}")

# Load correct graph template
template_file = os.path.join(DATA_DIR, f'processed_graph_sim_HD{hd_val}_001.pt')
graph = torch.load(template_file, weights_only=False).to(DEVICE)
print(f"  Using HD{hd_val} template → {graph['fine'].pos.shape[0]:,} fine nodes")

# ─────────────────────────────────────────────────
#  LOAD GROUND TRUTH + INFERENCE
# ─────────────────────────────────────────────────
ds = torch.load(val_path, weights_only=False)
X_scaled = ds.X.squeeze(0).to(DEVICE)      # [N, 16]
T_scaled_gt = ds.T.squeeze(0).to(DEVICE)   # [N, 5]

graph['fine'].x = X_scaled

print("Running inference...")
with torch.no_grad():
    pred_scaled = model(graph).cpu().numpy()

# Inverse transform (exact same as inference.py)
t_real_gt   = scalers['temp'].inverse_transform(T_scaled_gt[:, 0:1].cpu().numpy())
p_real_gt   = inverse_log_transform_pressure(
    scalers['pressure'].inverse_transform(T_scaled_gt[:, 1:2].cpu().numpy())
)
uvw_gt      = scalers['vel'].inverse_transform(T_scaled_gt[:, 2:5].cpu().numpy())

t_real_pred = scalers['temp'].inverse_transform(pred_scaled[:, 0:1])
p_real_pred = inverse_log_transform_pressure(
    scalers['pressure'].inverse_transform(pred_scaled[:, 1:2])
)
uvw_pred    = scalers['vel'].inverse_transform(pred_scaled[:, 2:5])

coords = graph['fine'].pos.cpu().numpy()
xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]

vel_mag_gt   = np.sqrt(uvw_gt[:,0]**2   + uvw_gt[:,1]**2   + uvw_gt[:,2]**2)
vel_mag_pred = np.sqrt(uvw_pred[:,0]**2 + uvw_pred[:,1]**2 + uvw_pred[:,2]**2)

err_T    = t_real_pred.flatten() - t_real_gt.flatten()
err_P    = p_real_pred.flatten() - p_real_gt.flatten()
err_Vmag = vel_mag_pred - vel_mag_gt

# ─────────────────────────────────────────────────
#  METRICS (focus on stagnation / wall)
# ─────────────────────────────────────────────────
wall_mask       = ys < 1e-4
stagnation_mask = wall_mask & (np.abs(xs - GEOM['jet_center_x']) < 0.005) & \
                  (np.abs(zs - GEOM['jet_center_z']) < 0.005)

print("\n" + "="*70)
print("DIAGNOSTIC RESULTS (v2 model)")
print("="*70)
print(f"Peak Temperature Error          : {t_real_pred.max() - t_real_gt.max():+.2f} °C")
print(f"MAE Temperature (wall)          : {np.abs(err_T[wall_mask]).mean():.2f} °C")
print(f"MAE Temperature (stagnation)    : {np.abs(err_T[stagnation_mask]).mean():.2f} °C")
print(f"MAE Pressure                    : {np.abs(err_P).mean():.0f} Pa")
print(f"MAE Velocity Magnitude          : {np.abs(err_Vmag).mean():.3f} m/s")

# ─────────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────────
plt.style.use('dark_background')

# 1. Wall temperature profile (centerline)
center_mask = wall_mask & (np.abs(zs - GEOM['jet_center_z']) < 0.001)
idx = np.argsort(xs[center_mask])

plt.figure(figsize=(11, 5))
plt.plot(xs[center_mask][idx], t_real_gt[center_mask][idx], 'r-', lw=2.5, label='CFD Ground Truth')
plt.plot(xs[center_mask][idx], t_real_pred[center_mask][idx], 'cyan', lw=2.5, label='Surrogate')
plt.axvline(GEOM['jet_center_x'], color='yellow', ls='--', alpha=0.8, label='Stagnation Point')
plt.title('Wall Temperature Profile — Centerline (Stagnation Zone)')
plt.xlabel('X coordinate (m)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('diagnostic_wall_profiles.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Temperature error map on bottom wall
plt.figure(figsize=(10, 6))
sc = plt.scatter(xs[wall_mask], zs[wall_mask], c=err_T[wall_mask],
                 s=3, cmap='RdBu_r', vmin=-10, vmax=10)
plt.colorbar(sc, label='ΔT (°C)  [Prediction − Ground Truth]')
plt.title('Temperature Error Map on Bottom Wall (y ≈ 0)')
plt.xlabel('X (m)'); plt.ylabel('Z (m)')
plt.axvline(GEOM['jet_center_x'], color='yellow', ls='--', alpha=0.7)
plt.tight_layout()
plt.savefig('diagnostic_error_map.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Save VTP for ParaView
cloud = pv.PolyData(coords)
cloud["T_pred (°C)"]      = t_real_pred.flatten()
cloud["T_gt (°C)"]        = t_real_gt.flatten()
cloud["T_error (°C)"]     = err_T
cloud["P_error (Pa)"]     = err_P
cloud["VelMag_error"]     = err_Vmag
cloud.save("diagnostic_comparison.vtp")

print("\nFiles saved:")
print("   • diagnostic_wall_profiles.png")
print("   • diagnostic_error_map.png")
print("   • diagnostic_comparison.vtp   ← open in ParaView")
print("\nDiagnosis complete. Please share the console output and describe the two plots (especially peak T error and whether the artificial spikes are gone).")