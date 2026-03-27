import os
import glob
import numpy as np

# --- CONFIGURATION ---
TENSOR_DIR = r"C:\Users\sudee\Desktop\work\cfd\ML_Tensors"

print(f"Scanning directory: {TENSOR_DIR}")
npz_files = glob.glob(os.path.join(TENSOR_DIR, "*.npz"))

if not npz_files:
    print("❌ No .npz files found! Check your path.")
    exit()

print(f"Found {len(npz_files)} tensor files. Calculating absolute global physics bounds...\n")

# Initialize global mins to infinity, and maxs to negative infinity
global_mins = np.full(6, np.inf)
global_maxs = np.full(6, -np.inf)

QUANTITIES = [
    "Temperature", 
    "Pressure", 
    "Turbulent_Kinetic_Energy", 
    "Velocity_X", 
    "Velocity_Y", 
    "Velocity_Z"
]

for idx, file_path in enumerate(npz_files, start=1):
    try:
        data = np.load(file_path)
        inputs = data['inputs']
        targets = data['targets']
        
        # The boolean mask of exactly where the fluid exists (ignores solid walls/empty space)
        fluid_mask = inputs[0] > 0.5 
        
        for c in range(6):
            # Extract ONLY the values that sit inside the fluid
            valid_physics_values = targets[c][fluid_mask]
            
            if len(valid_physics_values) > 0:
                local_min = valid_physics_values.min()
                local_max = valid_physics_values.max()
                
                # Update global trackers
                if local_min < global_mins[c]: global_mins[c] = local_min
                if local_max > global_maxs[c]: global_maxs[c] = local_max
                
    except Exception as e:
        print(f"🚨 Error reading {os.path.basename(file_path)}: {e}")

# --- PRINT THE EXACT DICTIONARY FOR PYTORCH ---
print("="*60)
print("✅ SCAN COMPLETE. COPY THIS EXACT BLOCK INTO digital_twin.py:")
print("="*60)
print("        self.global_bounds = {")
for c in range(6):
    is_last = "" if c == 5 else ","
    print(f"            {c}: [{global_mins[c]:.4f}, {global_maxs[c]:.4f}]{is_last}  # {QUANTITIES[c]}")
print("        }")
print("="*60)