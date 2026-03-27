import os
import glob
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Config
BASE_DIR = r"D:\data\JET"
SAVE_DIR = "raw_eda_results"
os.makedirs(SAVE_DIR, exist_ok=True)

def parse_simulation_params(filepath):
    basename = os.path.basename(filepath)
    vel_match = re.search(r"Vel_(-?\d+\.?\d*)", basename)
    pow_match = re.search(r"Pow_(\d+\.?\d*)", basename)
    velocity = abs(float(vel_match.group(1))) if vel_match else 5.0
    power = float(pow_match.group(1)) if pow_match else 50.0
    return velocity, power

def load_velocity_raw(grp):
    """Reuse the robust velocity loading logic for raw HDF5."""
    if "V_mag" in grp:
        return grp["V_mag"][:]
    elif "Velocity_Magnitude" in grp:
        return grp["Velocity_Magnitude"][:]
    elif "Velocity" in grp:
        raw = grp["Velocity"][:]
        if raw.ndim == 2 and raw.shape[1] >= 3:
            return np.linalg.norm(raw, axis=1)
        return raw.flatten()
    elif all(k in grp for k in ["Vx", "Vy", "Vz"]):
        vx, vy, vz = grp["Vx"][:], grp["Vy"][:], grp["Vz"][:]
        return np.sqrt(vx**2 + vy**2 + vz**2)
    else:
        return None

def run_raw_data_eda():
    print("🚀 Initializing Raw CFD Data EDA (H4 & H5)...", flush=True)
    
    all_files = glob.glob(os.path.join(BASE_DIR, "**", "*.h5"), recursive=True)
    print(f"Located {len(all_files)} HDF5 files.", flush=True)
    
    if not all_files:
        print("❌ Error: No .h5 files found in BASE_DIR.", flush=True)
        return

    # Data collectors for sampled nodes
    all_t_samples = []
    all_p_samples = []
    all_v_samples = []
    
    # Global ranges
    global_t_min, global_t_max = 1e10, -1e10
    global_p_min, global_p_max = 1e10, -1e10
    
    # Mesh audit (bounding boxes)
    mesh_bounds = []
    node_counts = []

    print("📊 Scanning HDF5 files and sampling nodes...", flush=True)
    for i, fp in enumerate(all_files):
        try:
            with h5py.File(fp, "r") as f:
                root_key = list(f.keys())[0]
                grp = f[root_key]
                
                coords = grp["Coordinates"][:]
                temp = grp["Temperature"][:]
                press = grp["Pressure"][:]
                vel = load_velocity_raw(grp)
                
                N = coords.shape[0]
                node_counts.append(N)
                
                # 1. Update Ranges
                global_t_min = min(global_t_min, temp.min())
                global_t_max = max(global_t_max, temp.max())
                global_p_min = min(global_p_min, press.min())
                global_p_max = max(global_p_max, press.max())
                
                # 2. Store Bounding Box [xmin, xmax, ymin, ymax, zmin, zmax]
                c_min, c_max = coords.min(axis=0), coords.max(axis=0)
                mesh_bounds.append(np.concatenate([c_min, c_max]))
                
                # 3. Subsample for histograms (50k nodes max per file)
                sample_n = min(N, 50000)
                idx = np.random.randint(0, N, sample_n)
                all_t_samples.append(temp[idx].flatten())
                all_p_samples.append(press[idx].flatten())
                if vel is not None:
                    all_v_samples.append(vel[idx].flatten())

            if (i+1) % 20 == 0:
                print(f"   Scanned {i+1}/{len(all_files)} files...", flush=True)
        except Exception as e:
            print(f"   ⚠️ Warning: Failed to process {fp}. Reason: {e}", flush=True)

    all_t_samples = np.concatenate(all_t_samples)
    all_p_samples = np.concatenate(all_p_samples)
    if all_v_samples:
        all_v_samples = np.concatenate(all_v_samples)

    print("\n📉 Generating Raw Data Plots...", flush=True)

    # --- Plot 1: Global Physical Distributions ---
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.hist(all_t_samples, bins=100, color='darkred', alpha=0.7)
    plt.title(f"Raw Temperature distribution\nRange: [{global_t_min:.1f}, {global_t_max:.1f}]")
    plt.xlabel("Temperature (K / °C)")

    plt.subplot(1, 3, 2)
    plt.hist(all_p_samples, bins=100, color='darkblue', alpha=0.7)
    plt.title(f"Raw Pressure distribution\nRange: [{global_p_min:.1f}, {global_p_max:.1f}]")
    plt.xlabel("Pressure (Pa)")

    plt.subplot(1, 3, 3)
    plt.hist(all_v_samples, bins=100, color='darkgreen', alpha=0.7)
    plt.title("Sampled Velocity Magnitude")
    plt.xlabel("Velocity (m/s)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "raw_physical_distributions.png"))
    plt.close()

    # --- Plot 2: Mesh Consistency (Bounding Boxes) ---
    mesh_bounds = np.array(mesh_bounds)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(mesh_bounds[:, 0], 'o-', label='xmin', color='red')
    plt.plot(mesh_bounds[:, 3], 'v-', label='xmax', color='darkred')
    plt.title("Mesh Spatial Bounds Consistency (X-axis)")
    plt.ylabel("Coordinate Val")
    plt.legend()
    plt.grid(True, alpha=0.2)

    plt.subplot(1, 2, 2)
    plt.plot(mesh_bounds[:, 2], 'o-', label='zmin', color='blue')
    plt.plot(mesh_bounds[:, 5], 'v-', label='zmax', color='darkblue')
    plt.title("Mesh Spatial Bounds Consistency (Z-axis)")
    plt.ylabel("Coordinate Val")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    plt.savefig(os.path.join(SAVE_DIR, "raw_mesh_alignment.png"))
    plt.close()

    # --- Plot 3: Node Density Distribution ---
    plt.figure(figsize=(8, 6))
    plt.hist(node_counts, bins=20, color='gray', edgecolor='black')
    plt.title("CFD Node Counts (Original Meshes)")
    plt.xlabel("Number of Nodes per Simulation")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(SAVE_DIR, "raw_node_counts.png"))
    plt.close()

    # --- Final Summary ---
    print("\n✅ Raw EDA Complete. Results saved in /raw_eda_results", flush=True)
    print(f"   Total Unique Nodes Sampled: {all_t_samples.shape[0]}")
    print(f"   Average Nodes per File: {np.mean(node_counts):.0f}")
    print(f"   Global Temp Range: [{global_t_min:.2f}, {global_t_max:.2f}]")
    print(f"   Global Press Range: [{global_p_min:.2f}, {global_p_max:.2f}]", flush=True)

if __name__ == "__main__":
    run_raw_data_eda()
