import os
import torch
import joblib
import numpy as np
import pandas as pd
import pyvista as pv
from LatentGNN import JetLatentGNN

# ==========================================
# 1. THE DIGITAL TWIN DASHBOARD (User Inputs)
# ==========================================
# Change these values to simulate any new scenario instantly!
USER_HD = 4.0               # H/D Ratio (Must be 4.0, 5.0, or 6.0)
USER_VELOCITY = 10.5        # Inlet Velocity (m/s)
USER_POWER = 55.0           # Heater Power (W)

DATA_DIR = r"D:\data\JET"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Physical Constants (Must match FeatureEngineering.py exactly)
rho, mu, k_fluid = 998.0, 0.001003, 0.6
D_m, A_chip_m2 = 0.00275, 30.25e-6
jet_center_x, jet_center_z = 0.05, 0.012
domain_x_max, domain_z_max = 0.10, 0.024

print("\n" + "="*50)
print(f"🚀 INITIATING DIGITAL TWIN INFERENCE")
print(f"   H/D: {USER_HD} | Vel: {USER_VELOCITY} m/s | Power: {USER_POWER} W")
print("="*50)

# ==========================================
# 2. LOAD SCALERS & TRAINED AI "BRAIN"
# ==========================================
print("\nLoading scalers and trained neural weights...")
scalers = {
    'global': joblib.load(os.path.join(DATA_DIR, 'scaler_global.pkl')),
    'spatial': joblib.load(os.path.join(DATA_DIR, 'scaler_spatial.pkl')),
    'skewed': joblib.load(os.path.join(DATA_DIR, 'scaler_skewed.pkl')),
    'temp': joblib.load(os.path.join(DATA_DIR, 'scaler_temp.pkl')),
    'vel': joblib.load(os.path.join(DATA_DIR, 'scaler_vel.pkl')),
    'target_power': joblib.load(os.path.join(DATA_DIR, 'scaler_target_power.pkl'))
}

# Initialize model with 12 features and load the best weights
model = JetLatentGNN(in_features=12, hidden_features=128, out_features=5).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(DATA_DIR, "best_jet_surrogate_model.pth"), weights_only=True))
model.eval()

# ==========================================
# 3. RECONSTRUCT THE PHYSICS STATE
# ==========================================
print("Retrieving geometric structural template...")
# Load the 100k-node structural skeleton for the requested H/D
template_file = os.path.join(DATA_DIR, f'processed_graph_sim_HD{int(USER_HD)}_001.pt')
graph = torch.load(template_file, weights_only=False).to(DEVICE)

# Extract raw X, Y, Z coordinates to rebuild the feature vectors
coords = graph['fine'].pos.cpu().numpy()  
xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]

print("Calculating boundary layer mechanics...")
# --- Calculate Physics Inputs ---
re_num = (rho * abs(USER_VELOCITY) * D_m) / mu
heat_flux = USER_POWER / A_chip_m2
# Bug 7 Fix: physically consistent Stanton number
dT_ref_proxy = max(50.0 - 20.0, 1e-6) # use sensible proxy for global max delta
q_star = heat_flux / (rho * cp_water * max(abs(USER_VELOCITY), 1e-6) * dT_ref_proxy)

radius = np.sqrt((xs - jet_center_x)**2 + (zs - jet_center_z)**2)
dist_outflow = np.minimum.reduce([xs, domain_x_max - xs, zs, domain_z_max - zs])
outlet_centers = np.array([[0.0, 0.012], [0.10, 0.012]])
dist_outlet = np.min([np.sqrt((xs - cx)**2 + (zs - cz)**2) for cx, cz in outlet_centers], axis=0)

delta_bl = D_m / np.sqrt(max(re_num, 1.0))
y_norm = ys / (delta_bl + 1e-10)
# Bug 5 Fix: Unified stagnation zone (radius < D_m)
stagnation_flag = (radius < D_m).astype(np.float32)

# --- Apply Scalers ---
g_raw = np.array([[USER_HD, np.log(max(re_num, 1.0)), heat_flux, q_star]] * len(coords))
s_raw = np.column_stack([xs, ys, zs, radius, dist_outflow, dist_outlet])
k_raw = y_norm.reshape(-1, 1) 
b_raw = stagnation_flag.reshape(-1, 1)

g_scaled = scalers['global'].transform(g_raw)
s_scaled = scalers['spatial'].transform(s_raw)
k_scaled = scalers['skewed'].transform(k_raw)
assert k_scaled.shape[1] == 1, "Skewed scaler expected 1 column (Y_norm only)"

# Assemble the final [100000, 12] input tensor and inject it into the graph
# 4 global + 6 spatial + 1 skewed + 1 binary = 12 ✓
X_inference = np.concatenate([g_scaled, s_scaled, k_scaled, b_raw], axis=1).astype(np.float32)
assert X_inference.shape[1] == 12, f"Inference feature count: {X_inference.shape[1]}, expected 12"
graph['fine'].x = torch.tensor(X_inference).to(DEVICE)

# ==========================================
# 4. EXECUTE SOLVER
# ==========================================
print("Solving momentum and thermal fields in Latent Space...")
with torch.no_grad():
    predictions_scaled = model(graph).cpu().numpy()

print("Decoding abstract tensors into real-world units...")
# Inverse transform to get back to °C, Pascal, and m/s
t_real = scalers['temp'].inverse_transform(predictions_scaled[:, 0:1])
p_real = scalers['target_power'].inverse_transform(predictions_scaled[:, 1:2])
uvw_real = scalers['vel'].inverse_transform(predictions_scaled[:, 2:5])

# ==========================================
# 5. PHYSICS ENFORCEMENT: HARD-LOCK INLET BOUNDARY
# ==========================================
print("Enforcing inlet boundary conditions...")
# Bug 5 Fix: Use full jet influence radius for boundary enforcement
inlet_nozzle_mask = (radius < (D_m)) & (ys > 0.010)
uvw_real[inlet_nozzle_mask, 0] = 0.0                # U (X-velocity) = 0
uvw_real[inlet_nozzle_mask, 1] = -abs(USER_VELOCITY) # V (Y-velocity) = exact user input
uvw_real[inlet_nozzle_mask, 2] = 0.0                # W (Z-velocity) = 0

# --- Bug 6 Fix: Enforce Outlet Boundary Conditions (Mass Continuity Proxy) ---
# Outlets at x=0 and x=0.10, radius D_m/2
D_outlet = D_m / 2
outlet_mask = (
    ((xs < D_outlet/2) | (xs > domain_x_max - D_outlet/2)) & 
    (np.abs(zs - 0.012) < D_outlet/2)
)
# Area-ratio velocity scaling (A_in * V_in = 2 * A_out * V_out)
# V_out = V_in * (D_in^2) / (2 * D_out^2)
v_out_scaled = abs(USER_VELOCITY) * (D_m**2) / (2 * D_outlet**2)

uvw_real[outlet_mask & (xs < 0.05), 0] = -v_out_scaled  # left outlet flow negative X
uvw_real[outlet_mask & (xs >= 0.05), 0] =  v_out_scaled  # right outlet flow positive X

n_locked_in = inlet_nozzle_mask.sum()
n_locked_out = outlet_mask.sum()
print(f" ✓ Locked {n_locked_in:,} inlet and {n_locked_out:,} outlet nodes.")

# ==========================================
# 6. EXPORT RESULTS
# ==========================================
csv_path = "jet_prediction_results.csv"
vtp_path = "jet_3d_digital_twin.vtp"  # Changed to .vtp for PolyData

print("\nPackaging Data Exports...")

# Derive absolute fluid Speed from 3D vectors
vel_magnitude = np.sqrt(uvw_real[:, 0]**2 + uvw_real[:, 1]**2 + uvw_real[:, 2]**2)

# Save CSV
df_out = pd.DataFrame({
    'X': xs, 'Y': ys, 'Z': zs,
    'Temperature_C': t_real.flatten(),
    'Pressure_Pa': p_real.flatten(),
    'U_vel': uvw_real[:, 0],
    'V_vel': uvw_real[:, 1],
    'W_vel': uvw_real[:, 2],
    'Velocity_Magnitude': vel_magnitude
})
df_out.to_csv(csv_path, index=False)
print(f" ✓ Quantitative Matrix Saved : {csv_path}")

# Save 3D Model (ParaView Compatible)
cloud = pv.PolyData(coords)
cloud["Temperature (°C)"] = t_real.flatten()
cloud["Pressure (Pa)"] = p_real.flatten()
cloud["Velocity Vector"] = uvw_real  
cloud["Velocity Magnitude (m/s)"] = vel_magnitude

# Save as .vtp instead of .vtu
cloud.save(vtp_path)
print(f" ✓ 3D Volumetric Mesh Saved  : {vtp_path}")

print("\n🎉 Digital Twin Cycle Complete! Open the .vtp file in ParaView to analyze the flow.")