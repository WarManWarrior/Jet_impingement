import os
import glob
import torch
import numpy as np
import h5py
from torch_geometric.nn import knn_graph
from gnn_pipeline import ThermalGNN, StandardScaler, MinMaxScaler, parse_simulation_params

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

@torch.no_grad()
def evaluate_full_mesh_error(filepath, model, x_scaler, y_scaler, device):
    """
    GPU-accelerated chunked inference on the full 238K mesh.
    Splits into ~30K-node subgraphs to stay under 6GB VRAM.
    """
    CHUNK_SIZE = 30000  # nodes per chunk — safe for 6GB GPU with hidden=64
    
    with h5py.File(filepath, "r") as f:
        group_name = list(f.keys())[0]
        grp = f[group_name]
        
        coords = torch.tensor(grp["Coordinates"][:], dtype=torch.float32)
        temp = torch.tensor(grp["Temperature"][:], dtype=torch.float32).view(-1, 1)
        pressure = torch.tensor(grp["Pressure"][:], dtype=torch.float32).view(-1, 1)
        v_mag = torch.tensor(grp["V_mag"][:], dtype=torch.float32).view(-1, 1)
        
        raw_true_y = torch.cat([temp, pressure, v_mag], dim=1)

    num_nodes = coords.shape[0]
    vel_norm, power_norm = parse_simulation_params(filepath)
    global_tensor = torch.tensor([vel_norm, vel_norm**2, power_norm], dtype=torch.float32).repeat(num_nodes, 1)
    
    centroid = coords.mean(dim=0)
    dist_to_center = torch.norm(coords - centroid, dim=1).view(-1, 1)
    
    eps = 1e-4
    boundary_flag = (
        (coords[:, 0] <= coords[:, 0].min() + eps) | (coords[:, 0] >= coords[:, 0].max() - eps) |
        (coords[:, 1] <= coords[:, 1].min() + eps) | (coords[:, 1] >= coords[:, 1].max() - eps) |
        (coords[:, 2] <= coords[:, 2].min() + eps) | (coords[:, 2] >= coords[:, 2].max() - eps)
    ).float().view(-1, 1)
    
    x_full = torch.cat([coords, dist_to_center, boundary_flag, global_tensor], dim=1)
    x_norm_full = x_scaler.transform(x_full)
    
    # Chunked GPU inference: split nodes, build local KNN per chunk, forward pass, concatenate
    num_chunks = (num_nodes + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"-> Chunked GPU inference: {num_nodes} nodes in {num_chunks} chunks of ~{CHUNK_SIZE}")
    
    all_preds = []
    model = model.to(device)
    
    for i in range(num_chunks):
        start = i * CHUNK_SIZE
        end = min((i + 1) * CHUNK_SIZE, num_nodes)
        
        chunk_coords = coords[start:end].to(device)
        chunk_x = x_norm_full[start:end].to(device)
        
        # Build local KNN graph for this chunk
        chunk_edge_index = knn_graph(chunk_coords, k=6, loop=False)
        row, col = chunk_edge_index
        displacements = chunk_coords[row] - chunk_coords[col]
        distances = torch.norm(displacements, p=2, dim=1).view(-1, 1)
        chunk_edge_attr = torch.cat([distances, displacements], dim=1)
        
        # Forward pass on GPU with mixed precision
        with torch.amp.autocast('cuda'):
            chunk_pred = model(chunk_x, chunk_edge_index, chunk_edge_attr)
        
        all_preds.append(chunk_pred.cpu())
        
        # Free GPU memory between chunks
        del chunk_coords, chunk_x, chunk_edge_index, chunk_edge_attr, chunk_pred
        torch.cuda.empty_cache()
        
        print(f"   Chunk {i+1}/{num_chunks} done ({end-start} nodes)")
    
    pred_norm = torch.cat(all_preds, dim=0)
    pred_raw = y_scaler.inverse_transform(pred_norm)
    
    # Composite error normalized by target std
    raw_error = torch.abs(pred_raw - raw_true_y)
    normed_error = raw_error / (raw_true_y.std(dim=0, keepdim=True) + 1e-5)
    composite_error = normed_error.sum(dim=1)
    
    return composite_error.numpy(), coords, temp, pressure, v_mag, group_name

def adaptive_sampling_pipeline(val_dir, train_dir, out_dir, target_extra_nodes=10000):
    """
    Evaluates GNN against High Fidelity validations sets.
    Finds hardest structural components, appending exactly those coordinates identically back into an upgraded Structural logic.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Init Adaptive PINN Sampling Logic Target: {device} | Extra Nodes: {target_extra_nodes}")
    
    model, x_scale, y_scale = load_surrogate_pipeline(device=device)
    
    val_files = glob.glob(os.path.join(val_dir, "**", "*.h5"), recursive=True)
    if not val_files:
        raise FileNotFoundError(f"Missing High-Fidelity Validation bounds under {val_dir}")
        
    os.makedirs(out_dir, exist_ok=True)
    
    # Extract the absolute hardest tracking limits logically from universally sweeping validation configurations dynamically
    for v_file in val_files:
        print(f"\n--- Sweeping CFD Error Manifold: {os.path.basename(v_file)} ---")
        errors, c_coords, c_t, c_p, c_v, grp_name = evaluate_full_mesh_error(v_file, model, x_scale, y_scale, device)
        
        # Identify top N error nodes
        hard_indices = np.argsort(errors)[-target_extra_nodes:]
        print(f"Ranked and mathematically isolated Top {target_extra_nodes} PDE violation coordinates.")
        
        # Load the mapped 20K existing bounds to merge dynamically
        train_equivalent = os.path.basename(v_file)
        matching_train_path = glob.glob(os.path.join(train_dir, "**", train_equivalent), recursive=True)
        
        if not matching_train_path:
            print(f"  -> Original train match missing for {train_equivalent}. Skipping merge limits natively.")
            continue
            
        train_pf = matching_train_path[0]
        with h5py.File(train_pf, 'r') as tf:
            t_grp = tf[list(tf.keys())[0]]
            base_coords = t_grp["Coordinates"][:]
            base_t = t_grp["Temperature"][:]
            base_p = t_grp["Pressure"][:]
            base_v = t_grp["V_mag"][:]
            
        # Neural Augmentation Assembly
        aug_coords = np.vstack((base_coords, c_coords.numpy()[hard_indices]))
        aug_t = np.vstack((base_t, c_t.numpy()[hard_indices]))
        aug_p = np.vstack((base_p, c_p.numpy()[hard_indices]))
        aug_v = np.vstack((base_v, c_v.numpy()[hard_indices]))
        
        # Ensure Uniqueness dynamically inside matrix mappings removing structural duplicates
        unique_coords, u_idx = np.unique(aug_coords, axis=0, return_index=True)
        final_coords = aug_coords[u_idx]
        final_t = aug_t[u_idx]
        final_p = aug_p[u_idx]
        final_v = aug_v[u_idx]
        
        out_path = os.path.join(out_dir, train_equivalent)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        with h5py.File(out_path, 'w') as out_f:
            og_grp = out_f.create_group(grp_name)
            og_grp.create_dataset("Coordinates", data=final_coords)
            og_grp.create_dataset("Temperature", data=final_t)
            og_grp.create_dataset("Pressure", data=final_p)
            og_grp.create_dataset("V_mag", data=final_v)
            
        print(f"  -> Augmented Data Map `{out_path}` successfully natively written containing {len(final_coords)} Structural Physics Nodes.")

if __name__ == "__main__":
    VAL_DIR = r"D:\data\JET\H4_reduced\Validation_Masked_238K"
    TRAIN_DIR = r"D:\data\JET\H4_reduced\Training_Clustered_20K"
    OUT_DIR = r"D:\data\JET\H4_reduced\Training_Adaptive_Phase2"
    
    adaptive_sampling_pipeline(VAL_DIR, TRAIN_DIR, OUT_DIR, target_extra_nodes=10000)
    print("\n✅ Stage 2 Extraction Complete. Point `gnn_pipeline.py DATA_PATH` identically to `Training_Adaptive_Phase2` and identically trigger retraining routines integrating explicit PDE Physics Bounds dynamically.")
