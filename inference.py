"""
=============================================================================
  DIGITAL TWIN INFERENCE — JET IMPINGEMENT SURROGATE

  FIXES matching FeatureEngineering.py:
    FIX A : domain_y_max = 0.027 (actual mesh height). Fixes signed_dist_wall
             and inlet node detection at inference time.
    FIX C : bc_velocity uses vel/VEL_MAX (not (vel-VEL_MIN)/(VEL_MAX-VEL_MIN)).
    FIX D : signed_dist_wall = domain_y_max - ys (distance from inlet plane).
=============================================================================
"""

import os
import torch
import joblib
import numpy as np
import pandas as pd
import pyvista as pv
from LatentGNN import JetLatentGNN
from FeatureEngineering import (
    RE_REGIME_THRESHOLD,
    inverse_log_transform_pressure,
    VEL_MAX,
    GEOM,
    STAGNATION_FLAG_IDX,
)

# ─────────────────────────────────────────────────
#  1. USER INPUTS
# ─────────────────────────────────────────────────
USER_HD       = 4.0
USER_VELOCITY = 4.0
USER_POWER    = 70.0

DATA_DIR = r"D:\data\JET"
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rho, mu, cp_fluid = 998.0, 0.001003, 4182.0
D_m          = GEOM['D_m']
A_chip_m2    = GEOM['A_chip_m2']
jet_center_x = GEOM['jet_center_x']   # 0.05238
jet_center_z = GEOM['jet_center_z']   # 0.012
domain_x_max = GEOM['domain_x_max']   # 0.10
domain_y_max = GEOM['domain_y_max']   # FIX A: 0.027 (was 0.02)
domain_z_max = GEOM['domain_z_max']   # 0.024
T_inlet      = GEOM['T_inlet']        # 20.0

print("\n" + "="*52)
print(f"  DIGITAL TWIN INFERENCE")
print(f"  H/D: {USER_HD} | Vel: {USER_VELOCITY} m/s | Power: {USER_POWER} W")
print("="*52)

# ─────────────────────────────────────────────────
#  2. LOAD SCALERS AND MODEL
# ─────────────────────────────────────────────────
print("\nLoading scalers and model weights...")
scalers = {
    'global'  : joblib.load(os.path.join(DATA_DIR, 'scaler_global.pkl')),
    'spatial' : joblib.load(os.path.join(DATA_DIR, 'scaler_spatial.pkl')),
    'skewed'  : joblib.load(os.path.join(DATA_DIR, 'scaler_skewed.pkl')),
    'temp'    : joblib.load(os.path.join(DATA_DIR, 'scaler_temp.pkl')),
    'vel'     : joblib.load(os.path.join(DATA_DIR, 'scaler_vel.pkl')),
    'pressure': joblib.load(os.path.join(DATA_DIR, 'scaler_pressure.pkl')),
}

t_max_global = float(np.load(os.path.join(DATA_DIR, 't_max_global.npy'))[0])
print(f"  t_max_global = {t_max_global:.2f} °C")

model = JetLatentGNN(in_features=16, hidden_features=128, out_features=5).to(DEVICE)
checkpoint = torch.load(os.path.join(DATA_DIR, "best_jet_surrogate_model.pth"), weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

# ─────────────────────────────────────────────────
#  3. BUILD FEATURE VECTOR  (16 features)
# ─────────────────────────────────────────────────
print("Loading graph template and computing features...")
template_file = os.path.join(DATA_DIR, f'processed_graph_sim_HD{int(USER_HD)}_001.pt')
graph  = torch.load(template_file, weights_only=False).to(DEVICE)
coords = graph['fine'].pos.cpu().numpy()
xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]

re_num    = (rho * abs(USER_VELOCITY) * D_m) / mu
heat_flux = USER_POWER / A_chip_m2
dT_ref    = max(t_max_global - T_inlet, 1.0)
q_star    = heat_flux / (rho * cp_fluid * max(abs(USER_VELOCITY), 1e-6) * dT_ref)

radius       = np.sqrt((xs - jet_center_x)**2 + (zs - jet_center_z)**2)
dist_outflow = np.minimum.reduce([xs, domain_x_max - xs, zs, domain_z_max - zs])
outlet_centers = np.array([[0.0, jet_center_z], [domain_x_max, jet_center_z]])
dist_outlet  = np.min([np.sqrt((xs - cx)**2 + (zs - cz)**2) for cx, cz in outlet_centers], axis=0)

delta_bl = D_m / np.sqrt(max(re_num, 1.0))
y_norm   = ys / (delta_bl + 1e-10)

# FIX D: signed_dist_wall = distance from inlet plane (top), 0 at inlet, max at chip
signed_dist_wall = (domain_y_max - ys).astype(np.float32)

stagnation_flag = (radius < D_m).astype(np.float32)
is_HD4          = np.ones(len(xs), dtype=np.float32) * float(USER_HD == 4.0)
is_HD6          = np.ones(len(xs), dtype=np.float32) * float(USER_HD == 6.0)
re_regime       = np.ones(len(xs), dtype=np.float32) * float(re_num > RE_REGIME_THRESHOLD)

# FIX C: bc_velocity uses vel/VEL_MAX — consistent with training
y_actual_max    = ys.max()
inlet_node_mask = (ys > y_actual_max - 0.002) & (radius < D_m / 2.0)
vel_norm        = abs(USER_VELOCITY) / VEL_MAX   # [0.25, 1.0] — always has signal
bc_velocity     = np.where(inlet_node_mask, vel_norm, 0.0).astype(np.float32)

print(f"  Inlet nodes with bc_velocity: {inlet_node_mask.sum():,}")
print(f"  bc_velocity value: {vel_norm:.4f}  (vel={abs(USER_VELOCITY):.1f}/VEL_MAX={VEL_MAX:.1f})")

# Apply scalers
GLOBAL_COLS  = ['Log_Reynolds', 'Heat_Flux', 'Stanton_Proxy']
SPATIAL_COLS = ['X', 'Y', 'Z', 'Radius', 'Dist_Outflow', 'Dist_Outlet', 'signed_dist_wall']

g_raw    = np.array([[np.log(max(re_num, 1.0)), heat_flux, q_star]] * len(coords))
s_raw    = np.column_stack([xs, ys, zs, radius, dist_outflow, dist_outlet, signed_dist_wall])
k_raw    = y_norm.reshape(-1, 1)
b_raw    = np.column_stack([stagnation_flag, is_HD4, is_HD6, re_regime, bc_velocity])

g_scaled = scalers['global'].transform(g_raw)
s_scaled = scalers['spatial'].transform(s_raw)
k_scaled = scalers['skewed'].transform(k_raw)

# 3G + 7S + 1K + 5B = 16
X_inference = np.concatenate([g_scaled, s_scaled, k_scaled, b_raw], axis=1).astype(np.float32)
assert X_inference.shape[1] == 16, f"Expected 16 features, got {X_inference.shape[1]}"
graph['fine'].x = torch.tensor(X_inference).to(DEVICE)
print(f"  Feature tensor: {X_inference.shape}  ✓")

# ─────────────────────────────────────────────────
#  4. SOLVE
# ─────────────────────────────────────────────────
print("Solving in latent space...")
with torch.no_grad():
    predictions_scaled = model(graph).cpu().numpy()

print("Inverse-transforming to physical units...")
t_real    = scalers['temp'].inverse_transform(predictions_scaled[:, 0:1])
p_log_inv = scalers['pressure'].inverse_transform(predictions_scaled[:, 1:2])
p_real    = inverse_log_transform_pressure(p_log_inv)
uvw_real  = scalers['vel'].inverse_transform(predictions_scaled[:, 2:5])

# ─────────────────────────────────────────────────
#  5. BOUNDARY CONDITIONS
# ─────────────────────────────────────────────────
print("Enforcing boundary conditions...")

inlet_bc_mask = (radius < D_m / 2.0) & (ys > y_actual_max - 0.002)
uvw_real[inlet_bc_mask, 0] = 0.0
uvw_real[inlet_bc_mask, 1] = -abs(USER_VELOCITY)
uvw_real[inlet_bc_mask, 2] = 0.0
print(f"  ✓ Inlet : {inlet_bc_mask.sum():,} nodes → Vy = {-abs(USER_VELOCITY):.2f} m/s")

D_outlet = D_m / 2.0
r_outlet = D_outlet / 2.0
A_inlet  = np.pi * (D_m / 2.0) ** 2
A_outlet = np.pi * r_outlet ** 2
V_outlet = (A_inlet * abs(USER_VELOCITY)) / (2.0 * A_outlet)

left_mask  = (xs < r_outlet) & (np.abs(zs - jet_center_z) < r_outlet)
right_mask = (xs > domain_x_max - r_outlet) & (np.abs(zs - jet_center_z) < r_outlet)
uvw_real[left_mask,  :] = [-V_outlet, 0.0, 0.0]
uvw_real[right_mask, :] = [ V_outlet, 0.0, 0.0]
print(f"  ✓ Left outlet : {left_mask.sum():,} nodes → Vx = {-V_outlet:.2f} m/s")
print(f"  ✓ Right outlet: {right_mask.sum():,} nodes → Vx = {+V_outlet:.2f} m/s")

# ─────────────────────────────────────────────────
#  6. SANITY CHECK
# ─────────────────────────────────────────────────
vel_mag = np.sqrt(uvw_real[:, 0]**2 + uvw_real[:, 1]**2 + uvw_real[:, 2]**2)
print(f"\nSanity check:")
print(f"  Temperature  : {t_real.min():.2f} – {t_real.max():.2f} °C  "
      f"(expected {T_inlet:.0f} – {t_max_global:.1f} °C)")
print(f"  Pressure     : {p_real.min():.0f} – {p_real.max():.0f} Pa")
print(f"  Speed        : {vel_mag.min():.2f} – {vel_mag.max():.2f} m/s  "
      f"(peak ≈ {abs(USER_VELOCITY):.1f} m/s)")

if t_real.max() < T_inlet + 1.0:
    print("  ⚠ WARNING: max temperature is barely above ambient — model may not have converged on hot nodes")
if t_real.max() > t_max_global * 1.5:
    print("  ⚠ WARNING: max temperature is unphysically high — check scaler consistency")

# ─────────────────────────────────────────────────
#  7. EXPORT
# ─────────────────────────────────────────────────
csv_path = "jet_prediction_results.csv"
vtp_path = "jet_3d_digital_twin.vtp"
print("\nExporting results...")

df_out = pd.DataFrame({
    'X': xs, 'Y': ys, 'Z': zs,
    'Temperature_C'    : t_real.flatten(),
    'Pressure_Pa'      : p_real.flatten(),
    'U_vel'            : uvw_real[:, 0],
    'V_vel'            : uvw_real[:, 1],
    'W_vel'            : uvw_real[:, 2],
    'Velocity_Magnitude': vel_mag,
})
df_out.to_csv(csv_path, index=False)
print(f"  ✓ CSV: {csv_path}")

cloud = pv.PolyData(coords)
cloud["Temperature (°C)"]  = t_real.flatten()
cloud["Pressure (Pa)"]      = p_real.flatten()
cloud["Velocity Vector"]    = uvw_real
cloud["Velocity Magnitude"] = vel_mag
cloud.save(vtp_path)
print(f"  ✓ VTP: {vtp_path}")
print("\n🎉 Inference complete.")