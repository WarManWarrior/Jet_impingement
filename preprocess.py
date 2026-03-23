import h5py
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
import torch
from sklearn.cluster import MiniBatchKMeans

HAS_TORCH = torch.cuda.is_available()
if HAS_TORCH:
    print(f"GPU Available: Yes (PyTorch CUDA)")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch CUDA not available. Pipeline requires GPU for optimal performance.")

try:
    import faiss
    HAS_FAISS = True
    print("FAISS Available: Yes (CPU Clustering acceleration)")
except ImportError:
    HAS_FAISS = False
    print("FAISS not available. Falling back to sklearn for clustering.")


def compute_global_importance(h5_files, device):
    """Pass 1: Computes the global variance and importance mask across all simulations using online updates."""
    print("--- PASS 1: Computing Global Variance ---")
    
    count = 0
    # Initialize running sums lazily when we know the number of nodes
    sum_temp = None
    sq_temp = None
    sum_p = None
    sq_p = None
    sum_tke = None
    sq_tke = None
    sum_vmag = None
    sq_vmag = None
    sum_devt = None
    sq_devt = None
    
    for h5_file in tqdm(h5_files, desc="Pass 1 Progress"):
        try:
            with h5py.File(h5_file, "r") as f:
                group_name = list(f.keys())[0]
                grp = f[group_name]

                velocity = grp["Velocity"][:]
                p = grp["Pressure"][:].reshape(-1)
                temp = grp["Temperature"][:].reshape(-1)
                tke = grp["Turbulent_Kinetic_Energy"][:].reshape(-1)
                
                # Move to GPU via from_numpy to prevent CPU-side memory copying
                velocity_gpu = torch.from_numpy(velocity).to(device, dtype=torch.float32)
                p_gpu = torch.from_numpy(p).to(device, dtype=torch.float32)
                temp_gpu = torch.from_numpy(temp).to(device, dtype=torch.float32)
                tke_gpu = torch.from_numpy(tke).to(device, dtype=torch.float32)
                
                # Features
                v_mag = torch.norm(velocity_gpu, dim=1)
                # Global Temperature Deviation Proxy
                # Note: This is an approximate proxy for local spatial gradient. Future upgrade: FAISS kNN gradient.
                dev_t = torch.abs(temp_gpu - temp_gpu.median())
                
                if sum_temp is None:
                    n_nodes = temp_gpu.shape[0]
                    # Allocate global tracking buffers
                    sum_temp = torch.zeros(n_nodes, dtype=torch.float32, device=device)
                    sq_temp = torch.zeros(n_nodes, dtype=torch.float32, device=device)
                    sum_p = torch.zeros(n_nodes, dtype=torch.float32, device=device)
                    sq_p = torch.zeros(n_nodes, dtype=torch.float32, device=device)
                    sum_tke = torch.zeros(n_nodes, dtype=torch.float32, device=device)
                    sq_tke = torch.zeros(n_nodes, dtype=torch.float32, device=device)
                    sum_vmag = torch.zeros(n_nodes, dtype=torch.float32, device=device)
                    sq_vmag = torch.zeros(n_nodes, dtype=torch.float32, device=device)
                    sum_devt = torch.zeros(n_nodes, dtype=torch.float32, device=device)
                    sq_devt = torch.zeros(n_nodes, dtype=torch.float32, device=device)
                
                # Accumulate online sums for numerical stability later
                sum_temp += temp_gpu
                sq_temp += temp_gpu**2
                sum_p += p_gpu
                sq_p += p_gpu**2
                sum_tke += tke_gpu
                sq_tke += tke_gpu**2
                sum_vmag += v_mag
                sq_vmag += v_mag**2
                sum_devt += dev_t
                sq_devt += dev_t**2
                
                count += 1
        except Exception as e:
            print(f"Error processing {h5_file} in Pass 1: {e}")

    if count == 0:
        raise ValueError("No valid HDF5 files processed in Pass 1.")

    # Compute stable hybrid variance: 0.5 * absolute + 0.5 * relative
    def compute_feature_variance(sum_tensor, sq_tensor, count_val):
        mean_val = sum_tensor / count_val
        var_val = torch.clamp((sq_tensor / count_val) - mean_val**2, min=0)
        
        # Absolute + Relative Combined Variance
        # User Feedback: Use std(original_feature) which equals sqrt(variance) for scaling
        std_val = torch.sqrt(var_val + 1e-12)
        denom = torch.abs(mean_val) + std_val
        relative_var = var_val / (denom + 1e-6)
        final_var = 0.5 * var_val + 0.5 * relative_var
        
        # User Fix: Normalize by mean to balance multi-physics selection perfectly
        final_var = final_var / (final_var.mean() + 1e-6)
            
        return final_var

    var_temp = compute_feature_variance(sum_temp, sq_temp, count)
    var_p = compute_feature_variance(sum_p, sq_p, count)
    
    # Optional Hard Clamp on Pressure to prevent extreme inflation
    var_p = torch.clamp(var_p, max=2.0)
    
    var_tke = compute_feature_variance(sum_tke, sq_tke, count)
    var_vmag = compute_feature_variance(sum_vmag, sq_vmag, count)
    var_devt = compute_feature_variance(sum_devt, sq_devt, count)
    
    # Diagnostic Logging to catch feature dominance issues early
    print("\nFeature Variance Diagnostic (Mean Normalized Contribution):")
    print(f"  Temp:     {var_temp.mean().item():.6f}")
    print(f"  Pressure: {var_p.mean().item():.6f}")
    print(f"  TKE:      {var_tke.mean().item():.6f}")
    print(f"  V_mag:    {var_vmag.mean().item():.6f}")
    print(f"  Dev_T:    {var_devt.mean().item():.6f}")
    
    # Global Importance: Hybrid Mean-Max pooling to prevent single-feature dominance drowning out others
    importance_mean = (0.3 * var_temp + 0.2 * var_p + 0.2 * var_tke + 0.2 * var_vmag + 0.1 * var_devt)
    importance_max, _ = torch.max(torch.stack([var_temp, var_p, var_tke, var_vmag, var_devt], dim=1), dim=1)
    importance = 0.5 * importance_mean + 0.5 * importance_max
    
    # Normalize globally before quantile thresholding to bound exactly [0, 1]
    if importance.max() > 0:
        importance /= importance.max()
        
    return importance


def compute_global_mask_and_clusters(h5_files, importance, target_nodes=50000):
    """Extracts base coordinates, applies the global mask, and computes shared cluster indices."""
    print("\n--- Computing Global Mask & Clustering ---")
    
    # 1. Generate Global MASK with Adaptive Thresholding
    n_nodes = importance.shape[0]
    if n_nodes > 300000:
        filter_threshold = 0.98
    elif n_nodes < 100000:
        filter_threshold = 0.95
    else:
        filter_threshold = 0.97
        
    print(f"Adaptive threshold chosen based on graph size ({n_nodes} nodes): {filter_threshold}")
    threshold = torch.quantile(importance, filter_threshold)
    mask = importance >= threshold
    mask_cpu = mask.cpu().numpy()
    
    print(f"Global nodes retained after filtering: {mask.sum().item()}")

    # 2. Get reference Coordinates from the first valid file
    coords = None
    for h5_file in h5_files:
        try:
            with h5py.File(h5_file, "r") as f:
                 group_name = list(f.keys())[0]
                 coords = f[group_name]["Coordinates"][:]
                 break
        except Exception:
            pass
            
    if coords is None:
        raise ValueError("Could not read Coordinates from any file to perform clustering.")
        
    coords_f = coords[mask_cpu]
    importance_f = importance[mask].cpu().numpy()
    
    # 3. Optional Clustering using [x, y, z, importance]
    
    if len(coords_f) <= target_nodes:
        print(f"Mask returned {len(coords_f)} nodes (<= target {target_nodes}). Skipping clustering entirely.")
        return mask_cpu, np.arange(len(coords_f))
        
    print(f"Clustering to {target_nodes} canonical nodes on CPU...")
    # Normalize coords and importance to ensure proper Euclidean distance scaling
    coords_norm = coords_f / (np.abs(coords_f).max(axis=0) + 1e-8)
    imp_norm = importance_f / (importance_f.max() + 1e-8)
    
    # Scale explicitly to bias spatial vs sensitivity
    alpha = 0.7  # tunable parameter for spatial weight proportion
    coords_scaled = coords_norm * alpha
    importance_scaled = imp_norm * (1 - alpha)
    
    features = np.column_stack((coords_scaled, importance_scaled))
    
    if HAS_FAISS:
        print("Using FAISS CPU K-Means for accelerated multidimensional clustering...")
        features_f32 = np.ascontiguousarray(features, dtype=np.float32)
        kmeans = faiss.Kmeans(d=features_f32.shape[1], k=target_nodes, niter=20, verbose=False, gpu=False)
        kmeans.train(features_f32)
        _, labels = kmeans.index.search(features_f32, 1)
        labels = labels.squeeze()
    else:
        print("Using sklearn MiniBatchKMeans...")
        kmeans = MiniBatchKMeans(n_clusters=target_nodes, batch_size=10000, random_state=42)
        labels = kmeans.fit_predict(features)
        
    # 4. Select Representative Nodes vectorization
    df = pd.DataFrame({
        "label": labels,
        "score": importance_f,
        "idx": np.arange(len(labels))
    })
    
    # Extract the highest-scoring physical node from each computed cluster
    best = df.loc[df.groupby("label")["score"].idxmax()]
    reduced_indices = best["idx"].values
    
    return mask_cpu, reduced_indices


def apply_global_mask_and_save(h5_files, base_path, output_base, mask_cpu, reduced_indices, importance, device):
    """Pass 2: Apply the identical structural mask to all files to generate GNN-ready topology."""
    print("\n--- PASS 2: Applying Mask and Saving Uniform Data ---")
    
    # Create unified dataset directories
    out_dir_validate = os.path.join(output_base, "Validation_Masked_238K")
    out_dir_training = os.path.join(output_base, "Training_Clustered_20K")
    os.makedirs(out_dir_validate, exist_ok=True)
    os.makedirs(out_dir_training, exist_ok=True)
    
    # Pre-extract the final importance array since it's identical for all simulations
    importance_cpu = importance.cpu().numpy()
    imp_filtered = importance_cpu[mask_cpu].reshape(-1, 1)
    imp_final = imp_filtered[reduced_indices].reshape(-1, 1)
    
    for h5_file in tqdm(h5_files, desc="Pass 2 Progress"):
        rel_path = os.path.relpath(h5_file, base_path)
        out_path_val = os.path.join(out_dir_validate, rel_path)
        out_path_train = os.path.join(out_dir_training, rel_path)
        
        try:
            with h5py.File(h5_file, "r") as f:
                group_name = list(f.keys())[0]
                grp = f[group_name]

                coords = grp["Coordinates"][:]
                velocity = grp["Velocity"][:]
                p = grp["Pressure"][:]
                temp = grp["Temperature"][:]
                tke = grp["Turbulent_Kinetic_Energy"][:]
                
            # Compute specific engineered features for this file using efficient zero-copy loading
            velocity_gpu = torch.from_numpy(velocity).to(device, dtype=torch.float32)
            temp_gpu = torch.from_numpy(temp.reshape(-1)).to(device, dtype=torch.float32)
            v_mag = torch.norm(velocity_gpu, dim=1).cpu().numpy().reshape(-1, 1)
            dev_t = torch.abs(temp_gpu - temp_gpu.median()).cpu().numpy().reshape(-1, 1)
            
            # Filter Layer (Validation Graph)
            coords_f = coords[mask_cpu]
            velocity_f = velocity[mask_cpu]
            p_f = p[mask_cpu]
            temp_f = temp[mask_cpu]
            tke_f = tke[mask_cpu]
            vmag_f = v_mag[mask_cpu]
            devt_f = dev_t[mask_cpu]
            
            # Reduction Layer (Training Graph)
            coords_final = coords_f[reduced_indices]
            velocity_final = velocity_f[reduced_indices]
            p_final = p_f[reduced_indices]
            temp_final = temp_f[reduced_indices]
            tke_final = tke_f[reduced_indices]
            vmag_final = vmag_f[reduced_indices]
            devt_final = devt_f[reduced_indices]
            
            # Security constraint checks before persisting array to disk
            assert len(coords_final) == len(p_final) == len(tke_final) == len(vmag_final) == len(imp_final), "CRITICAL: Extraction matrix length mismatch!"
            
            # Save Masked Validation Output
            os.makedirs(os.path.dirname(out_path_val), exist_ok=True)
            with h5py.File(out_path_val, "w") as f_out:
                grp_out = f_out.create_group(group_name)
                grp_out.create_dataset("Coordinates", data=coords_f)
                grp_out.create_dataset("Velocity", data=velocity_f)
                grp_out.create_dataset("Pressure", data=p_f)
                grp_out.create_dataset("Temperature", data=temp_f)
                grp_out.create_dataset("Turbulent_Kinetic_Energy", data=tke_f)
                grp_out.create_dataset("V_mag", data=vmag_f)
                grp_out.create_dataset("Dev_T", data=devt_f)
                grp_out.create_dataset("Global_Importance", data=imp_filtered)
            
            # Save Clustered Training Output
            os.makedirs(os.path.dirname(out_path_train), exist_ok=True)
            with h5py.File(out_path_train, "w") as f_out:
                grp_out = f_out.create_group(group_name)
                grp_out.create_dataset("Coordinates", data=coords_final)
                grp_out.create_dataset("Velocity", data=velocity_final)
                grp_out.create_dataset("Pressure", data=p_final)
                grp_out.create_dataset("Temperature", data=temp_final)
                grp_out.create_dataset("Turbulent_Kinetic_Energy", data=tke_final)
                
                # Appending engineered features
                grp_out.create_dataset("V_mag", data=vmag_final)
                grp_out.create_dataset("Dev_T", data=devt_final)
                grp_out.create_dataset("Global_Importance", data=imp_final)
                
        except Exception as e:
            print(f"Error saving {h5_file} in Pass 2: {e}")


def main():
    base_path = r"D:\data\JET\H4"
    output_base = r"D:\data\JET\H4_reduced"
    
    if not os.path.exists(base_path):
        print(f"Error: Base path {base_path} does not exist.")
        return

    # Aggregate all HDF5 workloads
    h5_files = []
    print(f"Scanning {base_path} for H5 files...")
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".h5"):
                h5_files.append(os.path.join(root, file))
    
    print(f"Found {len(h5_files)} H5 files. Launching global pipeline...")
    
    if not HAS_TORCH:
        raise RuntimeError("PyTorch CUDA is required for the global variance computational pipeline.")
        
    device = torch.device('cuda')
    
    # Validation: Compare coordinates of first two files verifying identical topological structure across dataset
    if len(h5_files) >= 2:
        print("Validating coordinate strict alignment across sampling bounds...")
        with h5py.File(h5_files[0], "r") as f1, h5py.File(h5_files[1], "r") as f2:
            c1 = f1[list(f1.keys())[0]]["Coordinates"][:]
            c2 = f2[list(f2.keys())[0]]["Coordinates"][:]
            assert np.allclose(c1, c2), "CRITICAL: Coordinates do not identically match between simulations! Processing halted."
    
    # === Pipeline Execution Engine ===
    start_time = time.time()
    
    # 1. Online Compute Global Importance Map
    importance = compute_global_importance(h5_files, device)
    
    # 2. Extract Shared Architecture Topology
    mask_cpu, reduced_indices = compute_global_mask_and_clusters(
        h5_files, importance, target_nodes=20000
    )
    
    # 3. Standardize and Render all Simulations
    apply_global_mask_and_save(h5_files, base_path, output_base, mask_cpu, reduced_indices, importance, device)
    
    print(f"\nUnified Processing Finished in {time.time() - start_time:.2f}s.")

if __name__ == "__main__":
    main()
