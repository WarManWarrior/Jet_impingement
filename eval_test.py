import os
import glob
import torch
import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import re
from FeatureEngineering import inverse_log_transform_pressure, GEOM, JetImpingementDataset
from LatentGNN import JetLatentGNN
import __main__
__main__.JetImpingementDataset = JetImpingementDataset

DATA_DIR = r"D:\data\JET"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load scalers
scalers = {
    'global': joblib.load(os.path.join(DATA_DIR, 'scaler_global.pkl')),
    'spatial': joblib.load(os.path.join(DATA_DIR, 'scaler_spatial.pkl')),
    'skewed': joblib.load(os.path.join(DATA_DIR, 'scaler_skewed.pkl')),
    'temp': joblib.load(os.path.join(DATA_DIR, 'scaler_temp.pkl')),
    'vel': joblib.load(os.path.join(DATA_DIR, 'scaler_vel.pkl')),
    'pressure': joblib.load(os.path.join(DATA_DIR, 'scaler_pressure.pkl')),
}

model = JetLatentGNN(in_features=18, hidden_features=128, out_features=5).to(DEVICE)
checkpoint = torch.load(os.path.join(DATA_DIR, "best_jet_surrogate_model.pth"), weights_only=False, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load graph templates
templates = {}
for hd in [4, 5, 6]:
    f = os.path.join(DATA_DIR, f'processed_graph_sim_HD{hd}_001.pt')
    if os.path.exists(f):
        templates[hd] = torch.load(f, weights_only=False).to(DEVICE)

test_files = glob.glob(os.path.join(DATA_DIR, "test_sims", "*.pt"))

all_t_pred, all_t_gt = [], []
all_p_pred, all_p_gt = [], []
all_u_pred, all_u_gt = [], []
all_v_pred, all_v_gt = [], []
all_w_pred, all_w_gt = [], []
all_vmag_pred, all_vmag_gt = [], []

max_t_chip = -1000.0
min_p_stag_err = []
vmag_peak_dist = []

# Using GEOM values
jet_center_x = GEOM['jet_center_x']
jet_center_z = GEOM['jet_center_z']
D_m = GEOM['D_m']

for idx, fpath in enumerate(test_files):
    hd_match = re.search(r'H(\d+)', fpath)
    # the test files don't have H/D in name... Oh wait!
    # They are named Vel_-10m_per_sec_Pow_45W...
    # Where does HD come from?
    ds = torch.load(fpath, weights_only=False)
    X_scaled_full = ds.X.squeeze(0).to(DEVICE)
    T_scaled_gt_full = ds.T.squeeze(0).to(DEVICE)
    
    is_hd4 = float(X_scaled_full[0, 14].item())
    is_hd6 = float(X_scaled_full[0, 15].item())
    
    if is_hd4 > 0.5: hd_val = 4
    elif is_hd6 > 0.5: hd_val = 6
    else: hd_val = 5
    
    graph = templates[hd_val]
    num_nodes = graph['fine'].pos.shape[0]
    
    X_scaled = X_scaled_full[:num_nodes]
    T_scaled_gt = T_scaled_gt_full[:num_nodes]
    
    # Feature 13 and 14 are is_HD4 and is_HD6 in 18-feature layout
    # g_scaled (4), s_scaled(8), k_scaled(1) -> 13
    # b_raw(5): [stagnation_flag, is_HD4, is_HD6, re_regime, bc_velocity]
    # So is_HD4 is index 14, is_HD6 is index 15
    is_hd4 = float(X_scaled[0, 14].item())
    is_hd6 = float(X_scaled[0, 15].item())
    
    if is_hd4 > 0.5: hd_val = 4
    elif is_hd6 > 0.5: hd_val = 6
    else: hd_val = 5
    
    graph = templates[hd_val]
    graph['fine'].x = X_scaled
    
    with torch.no_grad():
        pred_scaled = model(graph).cpu().numpy()
        
    t_real_gt = scalers['temp'].inverse_transform(T_scaled_gt[:, 0:1].cpu().numpy()).flatten()
    p_real_gt = inverse_log_transform_pressure(scalers['pressure'].inverse_transform(T_scaled_gt[:, 1:2].cpu().numpy())).flatten()
    uvw_gt = scalers['vel'].inverse_transform(T_scaled_gt[:, 2:5].cpu().numpy())
    
    t_real_pred = scalers['temp'].inverse_transform(pred_scaled[:, 0:1]).flatten()
    p_real_pred = inverse_log_transform_pressure(scalers['pressure'].inverse_transform(pred_scaled[:, 1:2])).flatten()
    uvw_pred = scalers['vel'].inverse_transform(pred_scaled[:, 2:5])
    
    vmag_gt = np.linalg.norm(uvw_gt, axis=1)
    vmag_pred = np.linalg.norm(uvw_pred, axis=1)
    
    all_t_gt.append(t_real_gt)
    all_t_pred.append(t_real_pred)
    all_p_gt.append(p_real_gt)
    all_p_pred.append(p_real_pred)
    all_u_gt.append(uvw_gt[:, 0])
    all_u_pred.append(uvw_pred[:, 0])
    all_v_gt.append(uvw_gt[:, 1])
    all_v_pred.append(uvw_pred[:, 1])
    all_w_gt.append(uvw_gt[:, 2])
    all_w_pred.append(uvw_pred[:, 2])
    all_vmag_gt.append(vmag_gt)
    all_vmag_pred.append(vmag_pred)
    
    # Notes calculation:
    coords = graph['fine'].pos.cpu().numpy()
    xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
    wall_mask = ys < 1e-4
    
    # Max on chip surface T
    if wall_mask.sum() > 0:
        max_t_chip = max(max_t_chip, t_real_pred[wall_mask].max() - t_real_gt[wall_mask].max())
    
    # Stagnation minimum within 1D of CFD
    stag_mask = wall_mask & (np.abs(xs - jet_center_x) < D_m) & (np.abs(zs - jet_center_z) < D_m)
    if stag_mask.sum() > 0:
        p_pred_stag = p_real_pred[stag_mask].min()
        p_gt_stag = p_real_gt[stag_mask].min()
        # actually, the note says "Stagnation minimum within 1D of CFD", so we might just check if min matches? 
        # For now, let's just output it.
        pass

all_t_gt = np.concatenate(all_t_gt)
all_t_pred = np.concatenate(all_t_pred)
all_p_gt = np.concatenate(all_p_gt)
all_p_pred = np.concatenate(all_p_pred)
all_u_gt = np.concatenate(all_u_gt)
all_u_pred = np.concatenate(all_u_pred)
all_v_gt = np.concatenate(all_v_gt)
all_v_pred = np.concatenate(all_v_pred)
all_w_gt = np.concatenate(all_w_gt)
all_w_pred = np.concatenate(all_w_pred)
all_vmag_gt = np.concatenate(all_vmag_gt)
all_vmag_pred = np.concatenate(all_vmag_pred)

print(f"Max T chip diff: {max_t_chip}")

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error

def metrics(gt, pred, name):
    mae = mean_absolute_error(gt, pred)
    medae = median_absolute_error(gt, pred)
    rmse = np.sqrt(mean_squared_error(gt, pred))
    max_err = np.max(np.abs(gt - pred))
    r2 = r2_score(gt, pred)
    print(f"{name:15s} MAE: {mae:8.2f}, MedAE: {medae:8.2f}, RMSE: {rmse:8.2f}, MaxErr: {max_err:10.2f}, R2: {r2:6.3f}")

print("-" * 80)

metrics(all_t_gt, all_t_pred, "Temperature")
metrics(all_p_gt, all_p_pred, "Pressure")
metrics(all_u_gt, all_u_pred, "V_x")
metrics(all_v_gt, all_v_pred, "V_y")
metrics(all_w_gt, all_w_pred, "V_z")
metrics(all_vmag_gt, all_vmag_pred, "Velocity Magnitude")

