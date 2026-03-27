import h5py
import numpy as np
import time
import os
import glob
import re

# --- 1. CONFIGURATION ---
BASE_DIR = r"C:\Users\sudee\Desktop\work\cfd\h4"
TENSOR_OUT_DIR = r"C:\Users\sudee\Desktop\work\cfd\ML_Tensors"

os.makedirs(TENSOR_OUT_DIR, exist_ok=True)
N = 128 # 128x128x128 Grid

QUANTITIES = ["Temperature", "Pressure", "Turbulent_Kinetic_Energy", "Velocity_X", "Velocity_Y", "Velocity_Z"]

# --- 2. FILE SEARCH & GLOBAL BOUNDING BOX ---
print(f"Scanning directory: {BASE_DIR}")
h5_files = glob.glob(os.path.join(BASE_DIR, "**", "*.h5"), recursive=True)

if not h5_files:
    print("❌ No .h5 files found! Check your directory path.")
    exit()

print(f"Found {len(h5_files)} completed trials. Calculating Global Bounding Box...")
global_x_min, global_y_min, global_z_min = float('inf'), float('inf'), float('inf')
global_x_max, global_y_max, global_z_max = float('-inf'), float('-inf'), float('-inf')

for file_path in h5_files:
    with h5py.File(file_path, 'r') as hdf:
        trial_name = list(hdf.keys())[0]
        if "Coordinates" in hdf[trial_name]:
            coords = hdf[trial_name]["Coordinates"][:]
            global_x_min, global_y_min, global_z_min = np.minimum([global_x_min, global_y_min, global_z_min], coords.min(axis=0))
            global_x_max, global_y_max, global_z_max = np.maximum([global_x_max, global_y_max, global_z_max], coords.max(axis=0))

# Buffer to prevent edge-overflow
epsilon = 1e-6
global_x_max += epsilon; global_y_max += epsilon; global_z_max += epsilon

# Compute universal logical bins
x_bins = np.linspace(global_x_min, global_x_max, N + 1)
y_bins = np.linspace(global_y_min, global_y_max, N + 1)
z_bins = np.linspace(global_z_min, global_z_max, N + 1)

# --- 3. MASS VOXELIZATION PIPELINE ---
print("\n" + "="*60)
print("🚀 BEGINNING MASS VOXELIZATION TO PyTorch TENSORS")
print("="*60)

for idx, file_path in enumerate(h5_files, start=1):
    trial_name = os.path.basename(file_path).replace(".h5", "")
    out_file = os.path.join(TENSOR_OUT_DIR, f"{trial_name}_Tensor.npz")
    
    if os.path.exists(out_file):
        print(f"⏭️ [{idx}/{len(h5_files)}] {trial_name} already voxelized. Skipping...")
        continue
        
    start_time = time.time()
    print(f"⚙️ [{idx}/{len(h5_files)}] Processing {trial_name}...")
    
    try:
        with h5py.File(file_path, 'r') as hdf:
            grp = hdf[trial_name]
            coords = grp["Coordinates"][:]
            
            # Extract fields
            fields_dict = {q: grp[q][:].flatten() for q in ["Temperature", "Pressure", "Turbulent_Kinetic_Energy"]}
            vel = grp["Velocity"][:]
            fields_dict["Velocity_X"], fields_dict["Velocity_Y"], fields_dict["Velocity_Z"] = vel[:, 0], vel[:, 1], vel[:, 2]
            
            # Extract parameters safely
            raw_vel = str(grp.attrs.get("velocity_input", "0"))
            raw_pow = str(grp.attrs.get("power_input", "0"))
            
            vel_match = re.search(r"[-+]?\d*\.?\d+", raw_vel)
            pow_match = re.search(r"[-+]?\d*\.?\d+", raw_pow)
            
            vel_in = float(vel_match.group()) if vel_match else 0.0
            pow_in = float(pow_match.group()) if pow_match else 0.0
            
        # --- ROBUST SPATIAL BINNING (CLIPPED) ---
        idx_x = np.clip(np.digitize(coords[:, 0], x_bins) - 1, 0, N - 1)
        idx_y = np.clip(np.digitize(coords[:, 1], y_bins) - 1, 0, N - 1)
        idx_z = np.clip(np.digitize(coords[:, 2], z_bins) - 1, 0, N - 1)
        
        flat_indices = np.ravel_multi_index((idx_x, idx_y, idx_z), dims=(N, N, N))
        voxel_counts = np.bincount(flat_indices, minlength=N**3)
        valid_voxels = voxel_counts > 0
        
        # --- BUILD INPUT TENSOR ---
        inputs_3d = np.zeros((3, N, N, N), dtype=np.float32)
        
        # Fluid Mask
        mask_flat = np.zeros(N**3, dtype=np.float32)
        mask_flat[valid_voxels] = 1.0
        inputs_3d[0] = mask_flat.reshape((N, N, N))
        
        # Inject Parameters only where fluid exists
        inputs_3d[1] = inputs_3d[0] * vel_in
        inputs_3d[2] = inputs_3d[0] * pow_in
        
        # --- BUILD TARGET TENSOR ---
        targets_3d = np.zeros((6, N, N, N), dtype=np.float32)
        
        for c, field_name in enumerate(QUANTITIES):
            voxel_sums = np.bincount(flat_indices, weights=fields_dict[field_name], minlength=N**3)
            voxel_means = np.zeros(N**3, dtype=np.float32)
            voxel_means[valid_voxels] = voxel_sums[valid_voxels] / voxel_counts[valid_voxels]
            
            # Explicitly mask out empty voxels to be perfectly 0.0
            channel_data = voxel_means.reshape((N, N, N))
            targets_3d[c] = channel_data * inputs_3d[0]
            
        # --- CRITICAL VALIDATION (FIREWALL) ---
        if np.isnan(inputs_3d).any():
            raise ValueError(f"NaN detected in inputs for {trial_name}")
        if np.isnan(targets_3d).any():
            raise ValueError(f"NaN detected in targets for {trial_name}")

        # --- SAVE ---
        np.savez_compressed(
            out_file, 
            inputs=inputs_3d,   
            targets=targets_3d  
        )
        
        print(f"   ✅ Saved {trial_name}_Tensor.npz in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        print(f"   ❌ Error processing {trial_name}: {e}")

print("\n🎉 ALL TRIALS VOXELIZED SUCCESSFULLY!")