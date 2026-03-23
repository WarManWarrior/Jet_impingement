import os
import glob
import re
import h5py
import time
import random
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, knn_graph

# =============================================================================
# 1. Dataset Construction & Preprocessing
# =============================================================================

def parse_simulation_params(filepath):
    """
    Extracts global boundary conditions (Velocity, Power) from the filepath.
    Expected format example: 'Vel_-3.0m_per_sec_Pow_40W'
    """
    filename = os.path.basename(os.path.dirname(filepath)) or os.path.basename(filepath)
    
    velocity = 3.0  
    power = 40.0    
    
    vel_match = re.search(r'Vel_[A-Za-z_]*([-\d\.]+)m', filename)
    pow_match = re.search(r'Pow_[A-Za-z_]*([\d\.]+)W', filename)
    
    if vel_match: velocity = abs(float(vel_match.group(1)))
    if pow_match: power = float(pow_match.group(1))
    
    # Mathematical Extrapolation Guards
    if not (3.0 <= velocity <= 10.0) or not (40.0 <= power <= 80.0):
        print(f"Warning: Parameters out of bound limit! Vel:{velocity}, Pow:{power}")
        
    # Min-Max Normalization Native Scaling
    vel_norm = (velocity - 3.0) / 7.0
    power_norm = (power - 40.0) / 40.0
        
    return [vel_norm, power_norm]


class ThermalDataset(Dataset):
    def __init__(self, file_list, k_neighbors=12, transform=None, pre_transform=None):
        """
        Dynamically loads GNN graphs preventing data-leakage.
        """
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)
        self.file_list = file_list
        self.k_neighbors = k_neighbors

    def len(self):
        return len(self.file_list)

    def get(self, idx):
        filepath = self.file_list[idx]
        
        with h5py.File(filepath, "r") as f:
            group_name = list(f.keys())[0]
            grp = f[group_name]
            
            # Load Geometry (Inputs)
            coords = torch.tensor(grp["Coordinates"][:], dtype=torch.float32)
            
            # CRITICAL FIX: Velocity, TKE, Importance are NOT loaded as inputs to prevent leakage
            
            # Load Targets (Outputs)
            temp = torch.tensor(grp["Temperature"][:], dtype=torch.float32).view(-1, 1)
            pressure = torch.tensor(grp["Pressure"][:], dtype=torch.float32).view(-1, 1)
            
        num_nodes = coords.shape[0]
        
        # Boundary / Global Parameters
        global_params = parse_simulation_params(filepath)
        global_tensor = torch.tensor(global_params, dtype=torch.float32).repeat(num_nodes, 1)
        
        # Engineered Geometric Features (Physics Boundary Markers)
        # 1. Distance to geometric center (Proxy for jet core radial distance)
        centroid = coords.mean(dim=0)
        dist_to_center = torch.norm(coords - centroid, dim=1).view(-1, 1)
        
        # 2. Boundary Flag (identifies walls, inlets, and outlets by bounding box proxy tracking)
        eps = 1e-4
        boundary_flag = (
            (coords[:, 0] <= coords[:, 0].min() + eps) | (coords[:, 0] >= coords[:, 0].max() - eps) |
            (coords[:, 1] <= coords[:, 1].min() + eps) | (coords[:, 1] >= coords[:, 1].max() - eps) |
            (coords[:, 2] <= coords[:, 2].min() + eps) | (coords[:, 2] >= coords[:, 2].max() - eps)
        ).float().view(-1, 1)
        
        # Compile final feature matrix X (Dim = 3 (coords) + 1 (dist) + 1 (bound) + 2 (global) = 7)
        x = torch.cat([coords, dist_to_center, boundary_flag, global_tensor], dim=1)
        
        # Compile target matrix Y (Dim = 2)
        y = torch.cat([temp, pressure], dim=1)
        
        # Graph Construction
        edge_index = knn_graph(coords, k=self.k_neighbors, loop=False)
        
        # Explicit Edge Feature Representation [distance, dx, dy, dz]
        row, col = edge_index
        displacements = coords[row] - coords[col]
        distances = torch.norm(displacements, p=2, dim=1).view(-1, 1)
        edge_attr = torch.cat([distances, displacements], dim=1) # Dim = 4
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=coords)
        return data

# =============================================================================
# 2. Dataset Normalizer Utility
# =============================================================================

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, tensor_list):
        """Fits normalizations globally to the explicit subset of files."""
        all_data = torch.cat(tensor_list, dim=0)
        self.mean = all_data.mean(dim=0, keepdim=True)
        self.std = all_data.std(dim=0, keepdim=True)
        self.std[self.std == 0] = 1.0  # Prevent ZeroDiv

    def transform(self, tensor):
        return (tensor - self.mean.to(tensor.device)) / self.std.to(tensor.device)
        
    def inverse_transform(self, tensor):
        return (tensor * self.std.to(tensor.device)) + self.mean.to(tensor.device)

# =============================================================================
# 3. Model Architecture (Edge-Aware NNConv)
# =============================================================================

class ThermalGNN(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden=128):
        """
        Physics-informed Surrogate architecture routing local spatial distances natively into 
        network filters mapping node derivatives continuously mapping boundary conditions.
        """
        super().__init__()
        
        # Edge parameter parsing mappings
        self.edge_mlp1 = nn.Sequential(
            nn.Linear(edge_dim, 32),
            nn.ReLU(),
            nn.Linear(32, in_channels * hidden)
        )
        self.edge_mlp2 = nn.Sequential(
            nn.Linear(edge_dim, 32),
            nn.ReLU(),
            nn.Linear(32, hidden * hidden)
        )
        self.edge_mlp3 = nn.Sequential(
            nn.Linear(edge_dim, 32),
            nn.ReLU(),
            nn.Linear(32, hidden * hidden)
        )

        self.conv1 = NNConv(in_channels, hidden, self.edge_mlp1, aggr='mean')
        self.conv2 = NNConv(hidden, hidden, self.edge_mlp2, aggr='mean')
        self.conv3 = NNConv(hidden, hidden, self.edge_mlp3, aggr='mean')

        self.lin = nn.Linear(hidden, 2)

    def forward(self, x, edge_index, edge_attr):
        x = torch.relu(self.conv1(x, edge_index, edge_attr))
        x = torch.relu(self.conv2(x, edge_index, edge_attr))
        x = torch.relu(self.conv3(x, edge_index, edge_attr))
        return self.lin(x)

# =============================================================================
# 4. Training Engine & Physics Constraints
# =============================================================================

def smoothness_loss(predictions, edge_index):
    """L_smooth = Σ ||T_i - T_j|| over connected edges"""
    row, col = edge_index
    diff = predictions[row] - predictions[col]
    return torch.mean(diff ** 2)

def train_thermal_surrogate(data_dir, epochs=100, batch_size=1, hidden_dim=32, lr=1e-3, lambda_smooth=0.1):
    print(f"\n🚀 Initializing Generalizable Physics-Aware GNN Surrogate")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Compute Device Bound: {device}")
    
    # 1. Dataset Discovery & Split Mapping
    raw_files = glob.glob(os.path.join(data_dir, "**", "*.h5"), recursive=True)
    if len(raw_files) == 0:
        print("CRITICAL: Empty HDF5 Directory.")
        return None
        
    # Extrapolation Guard Filter
    all_files = []
    for filepath in raw_files:
        vel_norm, pow_norm = parse_simulation_params(filepath)
        # Keep exclusively datasets existing completely within bounds [0.0, 1.0] dynamically preventing limits breaking boundaries explicitly
        if (0.0 <= vel_norm <= 1.0) and (0.0 <= pow_norm <= 1.0):
            all_files.append(filepath)
            
    print(f"Filtered invalid CFD domains: Preserved {len(all_files)}/{len(raw_files)} secure datasets.")
        
    random.seed(42)
    random.shuffle(all_files)
    
    # Structural 70/15/15 Folder Spilt Setup
    n_files = len(all_files)
    n_train = int(0.7 * n_files)
    n_val = int(0.15 * n_files)
    
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train+n_val]
    test_files = all_files[n_train+n_val:]
    
    print(f"Dataset Distribution: {n_files} Total Simulations")
    print(f"  -> Train: {len(train_files)} (70%)")
    print(f"  -> Val:   {len(val_files)} (15%)")
    print(f"  -> Test:  {len(test_files)} (15%)")
    
    train_dataset = ThermalDataset(train_files, k_neighbors=6)
    val_dataset = ThermalDataset(val_files, k_neighbors=6)
    test_dataset = ThermalDataset(test_files, k_neighbors=6)
    
    # 2. Dataset Normalizer Scaling Fitting
    print("\nFitting Normalizers Exclusively on Training Arrays...")
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    train_x_tensors = []
    train_y_tensors = []
    
    for i in range(len(train_dataset)):
        data = train_dataset[i]
        train_x_tensors.append(data.x)
        train_y_tensors.append(data.y)
        
    x_scaler.fit(train_x_tensors)
    y_scaler.fit(train_y_tensors)
    print("Standardization Params Locked.")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    assert train_dataset[0].x.shape[1] == 7, f"CRITICAL DIM MAPPING ERROR: X Tensor Shape={train_dataset[0].x.shape}"
    
    # 3. Model Initialization (In: 3 coords + 2 bounds + 2 global = 7 | Edge: 4 | Out: 2)
    model = ThermalGNN(in_channels=7, edge_dim=4, hidden=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # 4. Neural Optimization Phase
    print("\nStarting Neural Integration Engine")
    
    scaler_amp = GradScaler('cuda')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        start_time = time.time()
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Apply strict mathematical normalization (Scale down)
            x_norm = x_scaler.transform(batch.x)
            y_norm = y_scaler.transform(batch.y)
            
            with autocast('cuda'):
                # Formulate Predictions
                pred = model(x_norm, batch.edge_index, batch.edge_attr)
                
                # Target Separation
                t_pred, p_pred = pred[:, 0:1], pred[:, 1:2]
                t_true, p_true = y_norm[:, 0:1], y_norm[:, 1:2]
                
                # Fundamental Data Loss Framework
                loss_data = criterion(t_pred, t_true) + criterion(p_pred, p_true)
                
                # Advanced Laplacian Smoothness Constraints mapped mathematically identically across the Graph Edge topology
                loss_smooth_t = smoothness_loss(t_pred, batch.edge_index)
                loss_smooth_p = smoothness_loss(p_pred, batch.edge_index)
                loss_smooth = loss_smooth_t + loss_smooth_p
                
                # Full Convergent Aggregation
                loss = loss_data + (lambda_smooth * loss_smooth)
            
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            
            total_loss += loss.item()
            
        epoch_time = time.time() - start_time
        
        # Compute Validation Bounds 
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for v_batch in val_loader:
                    v_batch = v_batch.to(device)
                    v_x_norm = x_scaler.transform(v_batch.x)
                    v_y_norm = y_scaler.transform(v_batch.y)
                    
                    with autocast('cuda'):
                        v_pred = model(v_x_norm, v_batch.edge_index, v_batch.edge_attr)
                        vt_p, vp_p = v_pred[:, 0:1], v_pred[:, 1:2]
                        vt_t, vp_t = v_y_norm[:, 0:1], v_y_norm[:, 1:2]
                        
                        val_data_loss = criterion(vt_p, vt_t) + criterion(vp_p, vp_t)
                        val_smooth_t = smoothness_loss(vt_p, v_batch.edge_index)
                        val_smooth_p = smoothness_loss(vp_p, v_batch.edge_index)
                        val_smooth_loss = val_smooth_t + val_smooth_p
                        
                        val_loss += (val_data_loss + lambda_smooth * val_smooth_loss).item()
                    
            print(f"Epoch [{epoch+1:03d}/{epochs}] | "
                  f"Train Loss: {total_loss/len(train_loader):.4f} | "
                  f"Val Loss: {val_loss/len(val_loader):.4f} | "
                  f"Time: {epoch_time:.2f}s")
            
    print("\nPhase Complete. Surrogate Operator trained successfully.")
    
    # Structural Checkpoint Logic
    torch.save({
        'model_state': model.state_dict(),
        'x_mean': x_scaler.mean,
        'x_std': x_scaler.std,
        'y_mean': y_scaler.mean,
        'y_std': y_scaler.std
    }, "thermal_gnn.pth")
    print("Model physics parameters and scalers strictly written to 'thermal_gnn.pth'.")
    
    return model, x_scaler, y_scaler

# =============================================================================
# 5. Production Inference Logic
# =============================================================================

def execute_thermal_inference(model, x_scaler, y_scaler, base_graph_path, velocity, power, device='cpu'):
    """
    Deploys the real-time neural operator against an identical mesh extracting fully resolved 
    thermodynamic fields natively in milliseconds without invoking CFD solver arrays.
    """
    if not (3.0 <= velocity <= 10.0):
        raise ValueError(f"CRITICAL: User Velocity {velocity} breaches strict NN extrapolation bounds [3.0, 10.0]m/s.")
    if not (40.0 <= power <= 80.0):
        raise ValueError(f"CRITICAL: User Power {power} breaches strict NN extrapolation bounds [40.0, 80.0]W.")
        
    model.eval()
    
    # 1. Base Structure Loader
    with h5py.File(base_graph_path, "r") as f:
        grp = f[list(f.keys())[0]]
        coords = torch.tensor(grp["Coordinates"][:], dtype=torch.float32)
        
    # 2. Physics Geometric Marker Engines
    eps = 1e-4
    centroid = coords.mean(dim=0)
    dist_to_center = torch.norm(coords - centroid, dim=1).view(-1, 1)
    boundary_flag = (
        (coords[:, 0] <= coords[:, 0].min() + eps) | (coords[:, 0] >= coords[:, 0].max() - eps) |
        (coords[:, 1] <= coords[:, 1].min() + eps) | (coords[:, 1] >= coords[:, 1].max() - eps) |
        (coords[:, 2] <= coords[:, 2].min() + eps) | (coords[:, 2] >= coords[:, 2].max() - eps)
    ).float().view(-1, 1)
    
    # 3. Global Constraint Generation
    vel_norm = (velocity - 3.0) / 7.0
    pow_norm = (power - 40.0) / 40.0
    globals_tensor = torch.tensor([vel_norm, pow_norm], dtype=torch.float32).repeat(coords.shape[0], 1)
    
    # Spatial Graph Extractor
    edge_index = knn_graph(coords, k=12, loop=False)
    row, col = edge_index
    displacements = coords[row] - coords[col]
    distances = torch.norm(displacements, p=2, dim=1).view(-1, 1)
    edge_attr = torch.cat([distances, displacements], dim=1).to(device)

    # 4. Neural Operator Forward Pass
    x_matrix = torch.cat([coords, dist_to_center, boundary_flag, globals_tensor], dim=1).to(device)
    x_norm = x_scaler.transform(x_matrix)
    
    with torch.no_grad():
        edge_index = edge_index.to(device)
        pred_norm = model(x_norm, edge_index, edge_attr)
        
    # 5. Inverse Denormalization Space Mapping
    final_output = y_scaler.inverse_transform(pred_norm).cpu()
    return final_output[:, 0:1], final_output[:, 1:2]  # T, P

if __name__ == "__main__":
    DATA_PATH = r"D:\data\JET\H4_reduced\Training_Clustered_20K"
    train_thermal_surrogate(
        data_dir=DATA_PATH,
        epochs=100,
        batch_size=1
    )
