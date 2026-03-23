import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from gnn_pipeline import ThermalGNN, StandardScaler, execute_thermal_inference

def load_surrogate_pipeline(checkpoint_path="thermal_gnn.pth", device='cpu'):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Cannot find compiled model weights at {checkpoint_path}")
        
    print(f"Loading native Checkpoint arrays from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Recreate Architecture Base
    # Note: Using identical configuration parameters (channels=7, edge=4, hidden=32)
    model = ThermalGNN(in_channels=7, edge_dim=4, hidden=32).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # Reload Standard Scalers
    x_scaler = StandardScaler()
    x_scaler.mean = checkpoint['x_mean'].to(device)
    x_scaler.std = checkpoint['x_std'].to(device)
    
    y_scaler = StandardScaler()
    y_scaler.mean = checkpoint['y_mean'].to(device)
    y_scaler.std = checkpoint['y_std'].to(device)
    
    return model, x_scaler, y_scaler

def compute_metrics(pred_tensor, true_tensor):
    """Computes MAE and RMSE metrics securely over PyTorch arrays."""
    diff = pred_tensor - true_tensor
    mae = torch.abs(diff).mean().item()
    rmse = torch.sqrt((diff ** 2).mean()).item()
    return mae, rmse

def generate_interpolated_heatmap(coords, temp_preds, velocity, power, save_dir="visualizations", suffix="tempray"):
    """
    Renders 3D Spatial Heatmaps matching continuous thermal bounding limits natively.
    """
    os.makedirs(save_dir, exist_ok=True)
    coords_np = coords.numpy()
    temp_np = temp_preds.numpy()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Colormap standard tracking against Turbo gradient spectrum limit
    sc = ax.scatter(coords_np[:, 0], coords_np[:, 1], coords_np[:, 2], 
                    c=temp_np.squeeze(), cmap='turbo', s=5, alpha=0.8)
    
    cbar = plt.colorbar(sc, label='Target °C', pad=0.1)
    ax.set_title(f"Thermal Profile: Vel {velocity}m/s | Pow {power}W", fontsize=14, fontweight='bold')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    
    # Save Image to visual tracking system
    out_path = os.path.join(save_dir, f"inference_v{velocity}_p{power}_{suffix}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap physically rendered securely at '{out_path}'")
    plt.close()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Diagnostics Node Computing strictly across: {device}")
    
    # Step 1: Recover Models and Weights
    model, x_scale, y_scale = load_surrogate_pipeline(device=device)
    
    # Need exactly one valid underlying base spatial 20K graph coordinate file to structure inferences around
    base_data_path = r"D:\data\JET\H4_reduced\Training_Clustered_20K"
    try:
        available_files = glob.glob(os.path.join(base_data_path, "**", "*.h5"), recursive=True)
        base_h5 = available_files[0] 
        print(f"Sourced intrinsic geometry from: {base_h5}")
    except IndexError:
        raise FileNotFoundError("CRITICAL: Geometry source empty. Fix raw `base_data_path` pointer.")

    # Execute User Unseen Interpolation Parameters Arrays
    test_cases = [
        (6.3, 52),
        (7.8, 68),
        (5.5, 60)
    ]
    
    import h5py
    with h5py.File(base_h5, 'r') as f:
        grp = f[list(f.keys())[0]]
        coords_raw = torch.tensor(grp["Coordinates"][:], dtype=torch.float32)

    for vel, pow in test_cases:
        print(f"\n--- Predicting Interporlation Scope | Velocity {vel}m/s | Power {pow}W ---")
        
        # 1. Execute Inference Natively
        t_pred, p_pred = execute_thermal_inference(
            model=model, 
            x_scaler=x_scale, 
            y_scaler=y_scale, 
            base_graph_path=base_h5, 
            velocity=vel, 
            power=pow, 
            device=device
        )
        
        print(f"-> Local Thermal Ranges Identified  : MIN {t_pred.min().item():.2f}°C | MAX {t_pred.max().item():.2f}°C")
        print(f"-> Local Pressure Ranges Identified : MIN {p_pred.min().item():.2f}pa | MAX {p_pred.max().item():.2f}pa")
        
        # 2. Output Heatmap 3D Plots
        generate_interpolated_heatmap(coords_raw, t_pred, vel, pow, suffix="temp")
        generate_interpolated_heatmap(coords_raw, p_pred, vel, pow, suffix="pressure")
