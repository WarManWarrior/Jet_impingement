import h5py
import numpy as np
import time
import os

# --- 1. CONFIGURATION ---
TRIAL_PATH = r"C:\Users\sudee\Desktop\work\cfd\Vel_-3m_per_sec_Pow_40W.h5"
RESOLUTIONS = [64, 128, 192, 256]

# The fields we want to extract
QUANTITIES = ["Temperature", "Pressure", "Turbulent_Kinetic_Energy", "Velocity"]

print(f"Opening {os.path.basename(TRIAL_PATH)}...")
start_load = time.time()

try:
    with h5py.File(TRIAL_PATH, 'r') as hdf:
        trial_name = list(hdf.keys())[0]
        grp = hdf[trial_name]
        
        # 1. Load coordinates
        coords = grp["Coordinates"][:]
        
        # 2. Extract and format all physics fields into a dictionary
        fields_dict = {}
        for q_name in QUANTITIES:
            data = grp[q_name][:]
            
            # If the data is 3D (like Velocity), split it into X, Y, Z components
            if data.ndim > 1 and data.shape[1] == 3:
                fields_dict[f"{q_name}_X"] = data[:, 0].flatten()
                fields_dict[f"{q_name}_Y"] = data[:, 1].flatten()
                fields_dict[f"{q_name}_Z"] = data[:, 2].flatten()
            else:
                fields_dict[q_name] = data.flatten()
                
except Exception as e:
    print(f"❌ Failed to open file. Check path. Error: {e}")
    exit()

print(f"✅ Loaded {len(coords):,} points and {len(fields_dict)} physics fields in {time.time() - start_load:.2f} seconds.")

# --- 2. DEFINE THE PHYSICAL BOUNDING BOX ---
x_min, y_min, z_min = coords.min(axis=0)
x_max, y_max, z_max = coords.max(axis=0)

# Add a tiny buffer to the max bounds to prevent index overflow
epsilon = 1e-6
x_max += epsilon; y_max += epsilon; z_max += epsilon

print("\n" + "="*70)
print("🔬 MULTI-FIELD VOXELIZATION RESOLUTION TEST")
print("="*70)

# --- 3. THE HIGH-PERFORMANCE TEST LOOP ---
for N in RESOLUTIONS:
    print(f"\n🧱 Testing {N}x{N}x{N} Grid ({(N**3):,} total voxels)")
    start_calc = time.time()
    
    # --- A. COMPUTE THE SPATIAL BINS (Done ONCE per resolution) ---
    x_bins = np.linspace(x_min, x_max, N + 1)
    y_bins = np.linspace(y_min, y_max, N + 1)
    z_bins = np.linspace(z_min, z_max, N + 1)
    
    idx_x = np.digitize(coords[:, 0], x_bins) - 1
    idx_y = np.digitize(coords[:, 1], y_bins) - 1
    idx_z = np.digitize(coords[:, 2], z_bins) - 1
    
    flat_indices = np.ravel_multi_index((idx_x, idx_y, idx_z), dims=(N, N, N))
    
    # Calculate how many Ansys nodes fall into each voxel
    voxel_counts = np.bincount(flat_indices, minlength=N**3)
    valid_voxels = voxel_counts > 0
    
    print(f"   -> Voxels holding fluid: {np.sum(valid_voxels):,} / {(N**3):,}")
    
    # --- B. APPLY BINS TO EVERY PHYSICS FIELD ---
    for field_name, field_data in fields_dict.items():
        # Sum the values of the points inside each voxel
        voxel_sums = np.bincount(flat_indices, weights=field_data, minlength=N**3)
        
        # Calculate the mean for valid voxels
        voxel_means = np.zeros(N**3, dtype=np.float32)
        voxel_means[valid_voxels] = voxel_sums[valid_voxels] / voxel_counts[valid_voxels]
        
        # Map the voxel average back to the original 12M coordinates
        reconstructed_data = voxel_means[flat_indices]
        
        # Calculate errors
        absolute_errors = np.abs(field_data - reconstructed_data)
        mae = np.mean(absolute_errors)
        rmse = np.sqrt(np.mean(absolute_errors**2))
        max_err = np.max(absolute_errors)
        
        # Print results with aligned formatting
        print(f"      🔹 {field_name.ljust(25)} | MAE: {mae:<8.4f} | RMSE: {rmse:<8.4f} | Max Err: {max_err:.4f}")

    print(f"   ⏱️ Grid evaluation completed in {time.time() - start_calc:.2f} seconds.")

print("\n" + "="*70)
print("🏁 FULL TEST COMPLETE.")