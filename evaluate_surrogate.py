import os
import glob
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from gnn_pipeline import ThermalGNN, StandardScaler, MinMaxScaler, execute_thermal_inference

def load_surrogate_pipeline(checkpoint_path="thermal_gnn.pth", device='cpu'):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Cannot find compiled model weights at {checkpoint_path}")
        
    print(f"Loading native Checkpoint arrays from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = ThermalGNN(in_channels=8, edge_dim=4, hidden=64).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    x_scaler = StandardScaler()
    x_scaler.mean = checkpoint['x_mean'].to(device)
    x_scaler.std = checkpoint['x_std'].to(device)
    
    y_scaler = MinMaxScaler()
    y_scaler.min_val = checkpoint['y_min'].to(device)
    y_scaler.range_val = checkpoint['y_range'].to(device)
    
    return model, x_scaler, y_scaler

def compute_metrics(pred_tensor, true_tensor):
    """Computes MAE and RMSE metrics securely over PyTorch arrays."""
    diff = pred_tensor - true_tensor
    mae = torch.abs(diff).mean().item()
    rmse = torch.sqrt((diff ** 2).mean()).item()
    return mae, rmse

def generate_interactive_heatmap(coords, preds, velocity, power, save_dir="visualizations", suffix="temp"):
    """
    Renders an interactable 3D Spatial Heatmap using Plotly saving cleanly to an HTML browser view.
    """
    os.makedirs(save_dir, exist_ok=True)
    coords_np = coords.numpy()
    
    # Check if preds is already a numpy float or a pytorch tensor
    if torch.is_tensor(preds):
        preds_np = preds.numpy().squeeze()
    else:
        # e.g when passing a constant float for the velocity map
        preds_np = np.full(coords_np.shape[0], preds)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=coords_np[:, 0],
        y=coords_np[:, 1],
        z=coords_np[:, 2],
        mode='markers',
        marker=dict(
            size=2,          # Keep dots small to handle 20K arrays easily
            color=preds_np,
            colorscale='Turbo',
            colorbar=dict(title=f"Target {suffix}"),
            opacity=0.8
        )
    )])
    
    fig.update_layout(
        title=f"3D Interactive Surface Map | Velocity: {velocity:.3f} m/s | Power: {power:.3f} W | Layer: {suffix.upper()}",
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    outfile = os.path.join(save_dir, f"plotly_v{velocity}_p{power}_{suffix}.html")
    fig.write_html(outfile)
    print(f"-> Exported GUI Object: '{outfile}'")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Diagnostics Node Computing strictly across: {device}")
    
    # Step 1: Recover Models and Weights
    model, x_scale, y_scale = load_surrogate_pipeline(device=device)
    
    base_data_path = r"D:\data\JET\H4_reduced\Training_Clustered_20K"
    try:
        available_files = glob.glob(os.path.join(base_data_path, "**", "*.h5"), recursive=True)
        base_h5 = available_files[0] 
        print(f"Sourced intrinsic geometry from: {base_h5}")
    except IndexError:
        raise FileNotFoundError("CRITICAL: Geometry source empty. Fix raw `base_data_path` pointer.")

    # Execute User Unseen Interpolation Parameters Arrays
    test_cases = [
        (6.3, 52.0),
        (7.8, 68.0),
        (5.5, 60.0)
    ]
    
    import h5py
    with h5py.File(base_h5, 'r') as f:
        grp = f[list(f.keys())[0]]
        coords_raw = torch.tensor(grp["Coordinates"][:], dtype=torch.float32)

    # Logging CSV Memory State
    csv_log_path = "inference_results_max_limits.csv"
    results_list = []

    for vel, pow_w in test_cases:
        print(f"\n--- Predicting Interpolation Scope | Velocity {vel:.3f}m/s | Power {pow_w:.3f}W ---")
        
        # 1. Execute Inference Natively
        t_pred, p_pred, v_pred = execute_thermal_inference(
            model=model, 
            x_scaler=x_scale, 
            y_scaler=y_scale, 
            base_graph_path=base_h5, 
            velocity=vel, 
            power=pow_w, 
            device=device
        )
        
        t_max = t_pred.max().item()
        p_max = p_pred.max().item()
        v_max = v_pred.max().item()
        
        print(f"-> Thermal Max  : {t_max:.3f} °C")
        print(f"-> Pressure Max : {p_max:.3f} Pa")
        print(f"-> Velocity Target Max  : {v_max:.3f} m/s")
        
        results_list.append({
            'Target_Velocity_ms': f"{vel:.3f}",
            'Target_Power_W': f"{pow_w:.3f}",
            'Max_Temperature_C': f"{t_max:.3f}",
            'Max_Pressure_Pa': f"{p_max:.3f}",
            'Max_Velocity_ms': f"{v_max:.3f}"
        })
        
        # 1.5 Export Complete Spatial Node Array to CSV
        output_matrix = np.column_stack((
            coords_raw.numpy(),
            t_pred.numpy(),
            p_pred.numpy(),
            v_pred.numpy(),
            np.full(t_pred.shape[0], pow_w)
        ))
        
        header = "X,Y,Z,Temperature_C,Pressure_Pa,V_Mag_ms,Power_W_Input"
        full_csv_path = f"inference_v{vel:.3f}_p{pow_w:.3f}_nodes.csv"
        np.savetxt(full_csv_path, output_matrix, delimiter=",", header=header, comments="", fmt="%.3f")
        print(f"-> Full Spatial Matrix (~20K nodes) generated: '{full_csv_path}'")
        
        # 2. Output Plotly Interactive Browser Maps
        generate_interactive_heatmap(coords_raw, t_pred, vel, pow_w, suffix="Temperature")
        generate_interactive_heatmap(coords_raw, p_pred, vel, pow_w, suffix="Pressure")
        generate_interactive_heatmap(coords_raw, v_pred, vel, pow_w, suffix="Velocity_V_Mag")

    # Final Stage: Write CSV Record
    if results_list:
        with open(csv_log_path, 'w', newline='') as f:
            fieldnames = ['Target_Velocity_ms', 'Target_Power_W', 'Max_Temperature_C', 'Max_Pressure_Pa', 'Max_Velocity_ms']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_list)
        print(f"\n✅ All testing metrics successfully logged in quantitative target array at: {csv_log_path}")
