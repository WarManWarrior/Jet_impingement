import os
import glob
import re
import time
import torch
import h5py

# Global FNO Tensor Config
GRID_SIZE = 64
DTYPE = torch.float32
EPS = 1e-8

def parse_simulation_params(filepath):
    """Parses structural variables (Vel_XX_Pow_YY) from filename."""
    basename = os.path.basename(filepath)
    vel_match = re.search(r"Vel_(-?\d+\.?\d*)", basename)
    pow_match = re.search(r"Pow_(\d+\.?\d*)", basename)
    velocity = abs(float(vel_match.group(1))) if vel_match else 5.0
    power = float(pow_match.group(1)) if pow_match else 50.0
    return velocity, power

def extract_nozzle_diameter(filepath):
    """Maps H/D ratio from directory: H4 -> D=0.25, H5 -> D=0.20."""
    if "H4" in filepath.upper():
        return 1.0 / 4.0
    elif "H5" in filepath.upper():
        return 1.0 / 5.0
    else:
        raise ValueError(f"Cannot resolve H/D from path: {filepath}")

def load_velocity(grp, device):
    """Robust velocity loader — handles V_mag, Velocity (scalar or vector), Vx/Vy/Vz."""
    if "V_mag" in grp:
        return torch.tensor(grp["V_mag"][:], dtype=DTYPE).to(device).view(-1, 1)
    elif "Velocity_Magnitude" in grp:
        return torch.tensor(grp["Velocity_Magnitude"][:], dtype=DTYPE).to(device).view(-1, 1)
    elif "Velocity" in grp:
        raw = torch.tensor(grp["Velocity"][:], dtype=DTYPE).to(device)
        if raw.dim() == 2 and raw.shape[1] >= 3:
            return torch.norm(raw, dim=1, keepdim=True)
        return raw.view(-1, 1)
    elif all(k in grp for k in ["Vx", "Vy", "Vz"]):
        vx = torch.tensor(grp["Vx"][:], dtype=DTYPE).to(device)
        vy = torch.tensor(grp["Vy"][:], dtype=DTYPE).to(device)
        vz = torch.tensor(grp["Vz"][:], dtype=DTYPE).to(device)
        return torch.sqrt(vx**2 + vy**2 + vz**2).view(-1, 1)
    else:
        raise KeyError(f"No velocity field found. Keys: {list(grp.keys())}")

def voxelize(coords, values, vel_raw, grid_size=64):
    """
    Velocity-weighted voxelization (O(N)).
    Uses RAW velocity magnitude (not normalized) as aggregation weight.
    """
    idx = (coords * (grid_size - 1)).long().clamp(0, grid_size - 1)
    x, y, z = idx[:, 0], idx[:, 1], idx[:, 2]
    flat_idx = x * grid_size * grid_size + y * grid_size + z

    C = values.shape[1]
    device = coords.device

    # Weight by RAW velocity magnitude (normalized to prevent jet over-dominance)
    weight = vel_raw.abs()
    weight = weight / (weight.mean() + 1e-6)
    weight = torch.clamp(weight, max=5.0)
    weight = weight + 1e-3

    grid_sum = torch.zeros(grid_size**3, C, dtype=DTYPE, device=device)
    grid_weight = torch.zeros(grid_size**3, 1, dtype=DTYPE, device=device)
    grid_count = torch.zeros(grid_size**3, 1, dtype=DTYPE, device=device)

    grid_sum.index_add_(0, flat_idx, values * weight)
    grid_weight.index_add_(0, flat_idx, weight)
    grid_count.index_add_(0, flat_idx, torch.ones(flat_idx.shape[0], 1, dtype=DTYPE, device=device))

    grid_weight[grid_weight == 0] = 1.0
    grid = (grid_sum / grid_weight).view(grid_size, grid_size, grid_size, C)
    grid_count = grid_count.view(grid_size, grid_size, grid_size, 1)
    return grid, grid_count

# =============================================================================
# PASS 1 — Compute Global Normalization Stats (online Welford accumulator)
# =============================================================================

def compute_global_stats(all_files, device):
    """
    Single pass over all HDF5 files to compute global mean/std for T, P, V.
    Uses Welford's online algorithm to avoid loading everything into RAM.
    """
    print("\n📊 Pass 1: Computing GLOBAL normalization stats...")
    n = 0
    t_sum, t_sq = 0.0, 0.0
    p_sum, p_sq = 0.0, 0.0
    v_sum, v_sq = 0.0, 0.0

    for i, fp in enumerate(all_files):
        try:
            with h5py.File(fp, "r") as f:
                grp = f[list(f.keys())[0]]
                temp = torch.tensor(grp["Temperature"][:], dtype=DTYPE)
                press = torch.tensor(grp["Pressure"][:], dtype=DTYPE)
                vel = load_velocity(grp, torch.device('cpu'))

                temp = torch.clamp(temp, min=250, max=2000)
                press = torch.clamp(press, min=-1e6, max=1e6)

                count = temp.numel()
                n += count
                t_sum += temp.double().sum().item()
                t_sq += (temp.double() ** 2).sum().item()
                p_sum += press.double().sum().item()
                p_sq += (press.double() ** 2).sum().item()
                v_sum += vel.double().sum().item()
                v_sq += (vel.double() ** 2).sum().item()

            if (i + 1) % 20 == 0:
                print(f"   Scanned {i+1}/{len(all_files)} files...")
        except Exception as e:
            print(f"   ⚠️ Skipped {fp}: {e}")

    # Compute mean and std
    t_mean = t_sum / n
    p_mean = p_sum / n
    v_mean = v_sum / n
    t_std = (t_sq / n - t_mean ** 2) ** 0.5 + 1e-6
    p_std = (p_sq / n - p_mean ** 2) ** 0.5 + 1e-6
    v_std = (v_sq / n - v_mean ** 2) ** 0.5 + 1e-6

    stats = {
        "T_mean": t_mean, "T_std": t_std,
        "P_mean": p_mean, "P_std": p_std,
        "V_mean": v_mean, "V_std": v_std,
    }
    print(f"   Global Stats:")
    print(f"     T: mean={t_mean:.2f}, std={t_std:.2f}")
    print(f"     P: mean={p_mean:.2f}, std={p_std:.2f}")
    print(f"     V: mean={v_mean:.4f}, std={v_std:.4f}")
    return stats

# =============================================================================
# PASS 2 — Process each file using global stats
# =============================================================================

def process_single_h5_file(filepath, save_dir, global_stats, device):
    print(f"\nProcessing: {os.path.basename(filepath)}")
    start_time = time.time()

    try:
        device = torch.device(device) if isinstance(device, str) else device

        with h5py.File(filepath, "r") as f:
            grp = f[list(f.keys())[0]]
            coords = torch.tensor(grp["Coordinates"][:], dtype=DTYPE).to(device)
            temp = torch.tensor(grp["Temperature"][:], dtype=DTYPE).to(device).view(-1, 1)
            press = torch.tensor(grp["Pressure"][:], dtype=DTYPE).to(device).view(-1, 1)
            vel = load_velocity(grp, device)

        N = coords.shape[0]
        velocity, power = parse_simulation_params(filepath)
        D = extract_nozzle_diameter(filepath)

        # 1. Geometry normalization
        coords = coords / D
        min_c = coords.min(dim=0)[0]
        max_c = coords.max(dim=0)[0]
        coords = (coords - min_c) / (max_c - min_c + EPS)

        # 2. Clamp outliers
        temp = torch.clamp(temp, min=250, max=2000)
        press = torch.clamp(press, min=-1e6, max=1e6)

        # 3. Store RAW velocity for weighting + stagnation mask BEFORE normalization
        vel_raw = vel.clone()
        vel_raw = torch.clamp(vel_raw, min=0, max=vel_raw.quantile(0.99))  # remove spikes

        # 4. Normalize with GLOBAL stats
        temp_norm = (temp - global_stats["T_mean"]) / global_stats["T_std"]
        press_norm = (press - global_stats["P_mean"]) / global_stats["P_std"]
        vel_norm = (vel - global_stats["V_mean"]) / global_stats["V_std"]

        # 5. Voxelize normalized values (single pass, weighted by RAW velocity)
        values = torch.cat([temp_norm, press_norm, vel_norm], dim=1)
        voxel_output, grid_count = voxelize(coords, values, vel_raw, grid_size=GRID_SIZE)

        # 6. Stagnation mask from RAW velocity (voxelized cheaply without weights)
        idx = (coords * (GRID_SIZE - 1)).long().clamp(0, GRID_SIZE - 1)
        flat_idx = idx[:, 0] * GRID_SIZE**2 + idx[:, 1] * GRID_SIZE + idx[:, 2]
        vel_grid_sum = torch.zeros(GRID_SIZE**3, 1, dtype=DTYPE, device=device)
        vel_grid_cnt = torch.zeros(GRID_SIZE**3, 1, dtype=DTYPE, device=device)
        vel_grid_sum.index_add_(0, flat_idx, vel_raw)
        vel_grid_cnt.index_add_(0, flat_idx, torch.ones(N, 1, dtype=DTYPE, device=device))
        vel_grid_cnt[vel_grid_cnt == 0] = 1.0
        vel_raw_3d = (vel_grid_sum / vel_grid_cnt).view(GRID_SIZE, GRID_SIZE, GRID_SIZE)
        threshold = vel_raw_3d.mean() * 0.2  # robust: based on mean, not max
        stag_mask = (vel_raw_3d < threshold).float()
        
        # Density mask: helps model understand voxel coverage (smoothed)
        density_mask = (grid_count.squeeze(-1) > 0).float() + 1e-3

        # Normalize input parameters for scale consistency
        vel_norm_inp = velocity / 10.0
        pow_norm_inp = power / 100.0
        D_norm_inp = D / 0.5

        # Build 8-channel input: [x, y, z, vel_norm, pow_norm, D_norm, stagnation_mask, density_mask]
        lin = torch.linspace(0, 1, GRID_SIZE, device=device)
        x_m, y_m, z_m = torch.meshgrid(lin, lin, lin, indexing='ij')
        input_grid = torch.stack([
            x_m, y_m, z_m,
            torch.full_like(x_m, vel_norm_inp),
            torch.full_like(x_m, pow_norm_inp),
            torch.full_like(x_m, D_norm_inp),
            stag_mask,
            density_mask
        ], dim=0)

        # 8. Output grid [3, 64, 64, 64]
        output_grid = voxel_output.permute(3, 0, 1, 2)

        # 9. Validation
        density = (grid_count > 0).float().mean().item()
        print(f"  -> N={N} | Density: {density:.3f} | vel={velocity}, pow={power}, D={D:.3f}")
        print(f"  -> Output range: [{output_grid.min():.3f}, {output_grid.max():.3f}]")
        if density < 0.2:
            print(f"  -> ⚠️ LOW DENSITY")

        # 10. Save
        save_name = os.path.basename(filepath).replace(".h5", ".pt")
        save_path = os.path.join(save_dir, save_name)
        torch.save({
            "input": input_grid.cpu(),
            "output": output_grid.cpu(),
            "velocity": velocity, "power": power, "D": D
        }, save_path)

        del coords, temp, press, vel, vel_raw, values, voxel_output, grid_count, input_grid, output_grid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  -> Saved: {save_path} ({time.time() - start_time:.2f}s)")
        return True

    except Exception as e:
        print(f"  -> ❌ FAILED: {filepath}")
        print(f"     Reason: {e}")
        return False

# =============================================================================
# Main Pipeline
# =============================================================================

def execute_fno_pipeline(base_dir, output_parent_dir):
    print(f"\n🚀 FNO Voxelization Pipeline (2-Pass)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(output_parent_dir, exist_ok=True)

    all_files = glob.glob(os.path.join(base_dir, "**", "*.h5"), recursive=True)
    print(f"Found {len(all_files)} HDF5 files.")

    if not all_files:
        print("ERROR: No files found!")
        return

    # PASS 1: Global stats
    global_stats = compute_global_stats(all_files, device)

    # Save global scaler for inference
    scaler_path = os.path.join(output_parent_dir, "scaler.pt")
    torch.save(global_stats, scaler_path)
    print(f"\n💾 Global scaler saved to {scaler_path}")

    # PASS 2: Process each file
    print(f"\n📦 Pass 2: Voxelizing {len(all_files)} files...")
    success, fail = 0, 0
    for fw in all_files:
        if process_single_h5_file(fw, output_parent_dir, global_stats, device):
            success += 1
        else:
            fail += 1

    print(f"\n{'='*40}")
    print(f"FNO Pipeline Complete: {success}/{len(all_files)} (failed: {fail})")

if __name__ == "__main__":
    BASE_DIR = r"D:\data\JET"
    OUTPUT_DIR = r"D:\data\JET\FNO_Prepared"
    execute_fno_pipeline(BASE_DIR, OUTPUT_DIR)
