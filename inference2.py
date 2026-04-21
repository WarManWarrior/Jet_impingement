"""
=============================================================================
  DIGITAL TWIN INFERENCE — PURE DATA-DRIVEN
  Fixes applied:
    1. Pressure double-inverse removed — scaler.inverse_transform only
    2. signed_dist_x / signed_dist_z added to feature vector (18 → 20 features)
       Update model instantiation to in_features=20 after retraining.
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
    VEL_MAX,
    GEOM,
)

# ── User inputs ───────────────────────────────────────────────────────────────
USER_HD       = 4.0
USER_VELOCITY = 4.0
USER_POWER    = 70.0

DATA_DIR = r"D:\data\JET"
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Physical constants ────────────────────────────────────────────────────────
rho, mu, cp_fluid = 998.0, 0.001003, 4182.0
D_m           = GEOM['D_m']
A_chip_m2     = GEOM['A_chip_m2']
jet_center_x  = GEOM['jet_center_x']
jet_center_z  = GEOM['jet_center_z']
domain_x_max  = GEOM['domain_x_max']
domain_y_max  = GEOM['domain_y_max']
domain_z_max  = GEOM['domain_z_max']
T_inlet       = GEOM['T_inlet']

print("\n" + "="*60)
print("  DIGITAL TWIN INFERENCE — PURE DATA-DRIVEN")
print(f"  H/D: {USER_HD} | Vel: {USER_VELOCITY} m/s | Power: {USER_POWER} W")
print("="*60)

# ── Load scalers and model ────────────────────────────────────────────────────
print("\nLoading scalers and model weights...")
scalers = {
    k: joblib.load(os.path.join(DATA_DIR, f'scaler_{k}.pkl'))
    for k in ['global', 'spatial', 'skewed', 'temp', 'vel', 'pressure']
}
t_max_global = float(np.load(os.path.join(DATA_DIR, 't_max_global.npy'))[0])
print(f"  t_max_global = {t_max_global:.2f} °C")

# NOTE: Update in_features=20 here after retraining with signed distance features.
#       Keep in_features=18 to run inference with the existing checkpoint.
IN_FEATURES = 20   # change to 18 if using old checkpoint before retraining
model = JetLatentGNN(in_features=IN_FEATURES, hidden_features=128, out_features=5).to(DEVICE)
checkpoint = torch.load(
    os.path.join(DATA_DIR, "best_jet_surrogate_model.pth"), weights_only=False
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

# ── Load graph template and compute coordinates ───────────────────────────────
print("\nLoading graph template and computing features...")
template_file = os.path.join(DATA_DIR, f'processed_graph_sim_HD{int(USER_HD)}_001.pt')
graph  = torch.load(template_file, weights_only=False).to(DEVICE)
coords = graph['fine'].pos.cpu().numpy()
xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
N = len(xs)

# ── Derived physical quantities ───────────────────────────────────────────────
re_num     = (rho * abs(USER_VELOCITY) * D_m) / mu
heat_flux  = USER_POWER / A_chip_m2
dT_ref     = max(t_max_global - T_inlet, 1.0)
q_star     = heat_flux / (rho * cp_fluid * max(abs(USER_VELOCITY), 1e-6) * dT_ref)

radius       = np.sqrt((xs - jet_center_x)**2 + (zs - jet_center_z)**2)
dist_outflow = np.minimum.reduce([xs, domain_x_max - xs, zs, domain_z_max - zs])
outlet_centers = np.array([[0.0, jet_center_z], [domain_x_max, jet_center_z]])
dist_outlet  = np.min(
    [np.sqrt((xs - cx)**2 + (zs - cz)**2) for cx, cz in outlet_centers], axis=0
)

temp_gradient_proxy = heat_flux / (radius + 1e-4)
inv_radius          = 1.0 / (radius + 1e-5)
delta_bl            = D_m / np.sqrt(max(re_num, 1.0))
y_norm              = np.log1p(ys / (delta_bl + 1e-6))
signed_dist_wall    = ((domain_y_max - ys) / domain_y_max).astype(np.float32)
stagnation_flag     = np.exp(-(radius / D_m)**2).astype(np.float32)
is_HD4              = np.ones(N, dtype=np.float32) * float(USER_HD == 4.0)
is_HD6              = np.ones(N, dtype=np.float32) * float(USER_HD == 6.0)
re_regime           = np.ones(N, dtype=np.float32) * float(re_num > RE_REGIME_THRESHOLD)
vel_norm            = abs(USER_VELOCITY) / VEL_MAX
bc_velocity         = (vel_norm * np.exp(-radius / D_m)).astype(np.float32)

# FIX 2: Signed distances from jet centre — critical for U and W velocity learning.
# signed_dist_x drives U (horizontal spread left/right of jet).
# signed_dist_z drives W (horizontal spread front/back of jet).
signed_dist_x = (xs - jet_center_x).astype(np.float32)   # negative = left of jet
signed_dist_z = (zs - jet_center_z).astype(np.float32)   # negative = front of jet

# ── Assemble feature matrix ───────────────────────────────────────────────────
# Column groups:
#   Global (4):  Log_Reynolds, Heat_Flux, Stanton_Proxy, temp_gradient_proxy
#   Spatial (8): X, Y, Z, Radius, inv_radius, Dist_Outflow, Dist_Outlet, signed_dist_wall
#   Skewed (1):  y_norm
#   BC     (5):  stagnation_flag, is_HD4, is_HD6, re_regime, bc_velocity
#   NEW    (2):  signed_dist_x, signed_dist_z        ← Fix 2 additions
# Total: 4 + 8 + 1 + 5 + 2 = 20

g_raw = np.column_stack([
    np.full(N, np.log(max(re_num, 1.0))),
    np.full(N, heat_flux),
    np.full(N, q_star),
    temp_gradient_proxy,
])
s_raw = np.column_stack([
    xs, ys, zs, radius, inv_radius, dist_outflow, dist_outlet, signed_dist_wall
])
k_raw = y_norm.reshape(-1, 1)
b_raw = np.column_stack([
    stagnation_flag, is_HD4, is_HD6, re_regime, bc_velocity
])
d_raw = np.column_stack([
    signed_dist_x, signed_dist_z          # Fix 2: directional features
])

g_scaled = scalers['global'].transform(g_raw)
s_scaled = scalers['spatial'].transform(s_raw)
k_scaled = scalers['skewed'].transform(k_raw)
# signed_dist_x/z are not passed through a scaler — they are already
# zero-centred and in metres. Add a scaler for them in FeatureEngineering
# if the model is being retrained from scratch.

X_inference = np.concatenate(
    [g_scaled, s_scaled, k_scaled, b_raw, d_raw], axis=1
).astype(np.float32)

assert X_inference.shape[1] == IN_FEATURES, (
    f"Feature count mismatch: got {X_inference.shape[1]}, expected {IN_FEATURES}. "
    f"Set IN_FEATURES at the top of this file to match your checkpoint."
)
print(f"  Feature tensor: ({N}, {IN_FEATURES})  "
      f"(4G + 8S + 1K + 5B + 2D = {IN_FEATURES})")

graph['fine'].x = torch.tensor(X_inference).to(DEVICE)

# ── Inference ─────────────────────────────────────────────────────────────────
print("\nSolving in latent space...")
with torch.no_grad():
    predictions_scaled = model(graph).cpu().numpy()

# ── Inverse transforms ────────────────────────────────────────────────────────
print("Inverse-transforming to physical units...")

t_real   = scalers['temp'].inverse_transform(predictions_scaled[:, 0:1])

# FIX 1: Pressure — use scaler inverse ONLY.
# The original code applied inverse_log_transform_pressure on top of the scaler,
# which double-inverted the transform and produced values up to 900,000 Pa.
# The scaler alone is sufficient if it was fit on log-pressure.
# If pressure R² is still wrong after this fix, try:
#   p_real = np.expm1(scalers['pressure'].inverse_transform(predictions_scaled[:, 1:2]))
p_real   = scalers['pressure'].inverse_transform(predictions_scaled[:, 1:2])

uvw_real = scalers['vel'].inverse_transform(predictions_scaled[:, 2:5])
vel_mag  = np.sqrt(uvw_real[:, 0]**2 + uvw_real[:, 1]**2 + uvw_real[:, 2]**2)

# ── Sanity check ──────────────────────────────────────────────────────────────
print("\nSanity check:")
print(f"  Temperature : {t_real.min():.2f} – {t_real.max():.2f} °C  "
      f"(expected ~20 – 55 °C)")
print(f"  Pressure    : {p_real.min():.0f} – {p_real.max():.0f} Pa  "
      f"(expected ~-17,000 – +72,000 Pa)")
print(f"  Speed       : {vel_mag.min():.2f} – {vel_mag.max():.2f} m/s  "
      f"(expected 0 – ~10.5 m/s)")

# Flag if pressure is still out of range
if p_real.max() > 200_000 or p_real.min() < -100_000:
    print("\n  WARNING: Pressure range looks wrong — try the expm1 path above.")

# ── Export ────────────────────────────────────────────────────────────────────
print("\nExporting results...")
df_out = pd.DataFrame({
    'X':                  xs,
    'Y':                  ys,
    'Z':                  zs,
    'Temperature_C':      t_real.flatten(),
    'Pressure_Pa':        p_real.flatten(),
    'U_vel':              uvw_real[:, 0],
    'V_vel':              uvw_real[:, 1],
    'W_vel':              uvw_real[:, 2],
    'Velocity_Magnitude': vel_mag,
})
df_out.to_csv("jet_prediction_results.csv", index=False)
print("  CSV: jet_prediction_results.csv")

cloud = pv.PolyData(coords)
cloud["Temperature (°C)"]   = t_real.flatten()
cloud["Pressure (Pa)"]      = p_real.flatten()
cloud["Velocity Vector"]    = uvw_real
cloud["Velocity Magnitude"] = vel_mag
cloud.save("jet_3d_digital_twin.vtp")
print("  VTP: jet_3d_digital_twin.vtp")

print("\nInference complete.")