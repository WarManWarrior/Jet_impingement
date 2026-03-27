import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

# Global Config
DATA_DIR = r"D:\data\JET\FNO_Prepared"
SAVE_DIR = "eda_results"
os.makedirs(SAVE_DIR, exist_ok=True)

def run_comprehensive_eda():
    print("🚀 Initializing Comprehensive EDA Suite (Safe Mode)...", flush=True)
    
    files = glob.glob(os.path.join(DATA_DIR, "*.pt"))
    files = [f for f in files if "scaler.pt" not in os.path.basename(f)]
    print(f"Located {len(files)} processed matrices.", flush=True)
    
    if not files:
        print("❌ Error: No .pt files found.", flush=True)
        return

    # Data Accumulators
    all_t_samples = []
    all_p_samples = []
    all_v_samples = []
    
    peak_t_list = []
    peak_p_list = []
    vel_list = []
    pow_list = []
    D_list = []
    voxel_counts = []
    
    centerline_t = []
    centerline_p = []
    
    print("📊 Scanning files...", flush=True)
    for i, f in enumerate(files):
        try:
            data = torch.load(f)
            out_grid = data["output"]
            
            peak_t_list.append(out_grid[0].max().item())
            peak_p_list.append(out_grid[1].max().item())
            vel_list.append(data.get("velocity", 0.0))
            pow_list.append(data.get("power", 0.0))
            D_list.append(data.get("D", 0.0))
            
            # Density mask check
            density_mask = data["input"][7]
            voxel_counts.append(torch.sum(density_mask > 0.1).item())
            
            # Centerline (Z-profile)
            centerline_t.append(out_grid[0, 32, 32, :].numpy())
            centerline_p.append(out_grid[1, 32, 32, :].numpy())
            
            # Random samples for distribution
            flat_t = out_grid[0].flatten()
            flat_p = out_grid[1].flatten()
            flat_v = out_grid[2].flatten()
            
            sample_size = 500
            idx = np.random.randint(0, flat_t.numel(), sample_size)
            all_t_samples.extend(flat_t[idx].tolist())
            all_p_samples.extend(flat_p[idx].tolist())
            all_v_samples.extend(flat_v[idx].tolist())
            
            if (i+1) % 50 == 0:
                print(f"   Processed {i+1}/{len(files)}...", flush=True)
        except Exception as e:
            print(f"   ⚠️ Skipped {f}: {e}", flush=True)

    print("\n📈 Generating plots...", flush=True)
    
    # 1. Distributions
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.hist(all_t_samples, bins=50, color='red', alpha=0.7)
    plt.title("T_norm Distribution")
    plt.subplot(1, 3, 2)
    plt.hist(all_p_samples, bins=50, color='blue', alpha=0.7)
    plt.title("P_norm Distribution")
    plt.subplot(1, 3, 3)
    plt.hist(all_v_samples, bins=50, color='green', alpha=0.7)
    plt.title("V_norm Distribution")
    plt.savefig(os.path.join(SAVE_DIR, "global_distributions.png"))
    plt.close()

    # 2. Profiles
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    z_axis = np.linspace(0, 1, 64)
    plt.plot(z_axis, np.mean(centerline_t, axis=0), color='red')
    plt.title("Mean Temperature Centerline")
    plt.subplot(1, 2, 2)
    plt.plot(z_axis, np.mean(centerline_p, axis=0), color='blue')
    plt.title("Mean Pressure Centerline")
    plt.savefig(os.path.join(SAVE_DIR, "centerline_profiles.png"))
    plt.close()

    # 3. Trends
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(pow_list, peak_t_list, c=vel_list, alpha=0.6)
    plt.colorbar(label='Vel')
    plt.title("Peak T vs Power")
    plt.subplot(1, 2, 2)
    plt.scatter(vel_list, peak_p_list, c=pow_list, alpha=0.6)
    plt.colorbar(label='Pow')
    plt.title("Peak P vs Velocity")
    plt.savefig(os.path.join(SAVE_DIR, "parametric_trends.png"))
    plt.close()

    # 4. Diagnostics
    plt.figure(figsize=(8, 5))
    plt.hist(voxel_counts, bins=20, color='gray')
    plt.title("Voxel Coverage Diagnostics")
    plt.savefig(os.path.join(SAVE_DIR, "structural_diagnostics.png"))
    plt.close()

    print(f"✨ EDA Complete. Saved to {SAVE_DIR}/", flush=True)

if __name__ == "__main__":
    run_comprehensive_eda()
