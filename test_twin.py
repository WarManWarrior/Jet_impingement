import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from digital_twin import CFDUnet3D

# --- 1. CONFIGURATION ---
MODEL_PATH = r"saved_models\best_digital_twin.pth"
# We need to borrow the physical "shape" (the fluid mask) from one of your existing files.
# Pick any random .npz file from your ML_Tensors folder to act as the geometry template.
TEMPLATE_FILE = r"C:\Users\sudee\Desktop\work\cfd\ML_Tensors\Vel_-3m_per_sec_Pow_40W_Tensor.npz"

# 🔥 THE MAGIC NUMBERS: Ask the AI to predict an untested scenario!
TEST_VELOCITY = -7.5  # m/s
TEST_POWER = 62.0     # Watts

# Your exact global bounds for inverse scaling
GLOBAL_BOUNDS = {
    0: [20.0000, 61.7186],      # Temperature
    1: [-75042.7812, 452114.2500],  # Pressure
    2: [0.0000, 37.2057],       # TKE
    3: [-25.4716, 25.3899],     # Vel X
    4: [-10.3894, 10.6345],     # Vel Y
    5: [-9.1371, 9.0707]        # Vel Z
}

# --- 2. LOAD THE TRAINED BRAIN ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model onto {device}...")

model = CFDUnet3D().to(device)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()  # Lock the weights for prediction

# --- 3. PREPARE THE NEW INPUT DATA ---
print(f"Building custom tensor for Velocity: {TEST_VELOCITY} m/s | Power: {TEST_POWER} W")
template_data = np.load(TEMPLATE_FILE)
mask = template_data['inputs'][0]  # Extract the 128x128x128 fluid mask (1s and 0s)

# Build the 3-channel input tensor exactly like we did during training
new_inputs = np.zeros((3, 128, 128, 128), dtype=np.float32)
new_inputs[0] = mask
new_inputs[1] = mask * TEST_VELOCITY
new_inputs[2] = mask * TEST_POWER

# Convert to PyTorch tensor and add a "Batch" dimension at the front (Shape becomes 1, 3, 128, 128, 128)
input_tensor = torch.tensor(new_inputs).unsqueeze(0).to(device)

# --- 4. THE 10-MILLISECOND PREDICTION ---
print("Running AI Physics Prediction...")
with torch.no_grad():
    # This is the moment Ansys is replaced!
    raw_prediction = model(input_tensor)

# Pull the data off the GPU and remove the Batch dimension
prediction_np = raw_prediction.squeeze(0).cpu().numpy()

# --- 5. INVERSE SCALING (Back to real-world units) ---
print("Converting AI outputs back to °C and Pascals...")
real_physics = np.zeros_like(prediction_np)

for c in range(6):
    c_min, c_max = GLOBAL_BOUNDS[c]
    # Reverse the (val - min) / (max - min) formula
    real_physics[c] = (prediction_np[c] * (c_max - c_min)) + c_min
    # Re-apply the mask so solid walls drop exactly to 0.0, not ambient
    real_physics[c] = real_physics[c] * mask

# --- 6. VISUALIZATION (THE REWARD) ---
print("Plotting Top-Down Cross-Section...")
temp_3d = real_physics[0] # Channel 0 is Temperature

# 1. Clamp the AI's minor math errors so it doesn't drop below your 20C ambient
temp_3d = np.clip(temp_3d, 20.0, 100.0)

# 2. THE HEAT-SEEKER: Find the exact Z-height of the hottest fluid
# We multiply by the mask to ensure we only search inside the actual air/fluid
fluid_only_temp = temp_3d * mask
max_idx = np.unravel_index(np.argmax(fluid_only_temp), fluid_only_temp.shape)
hottest_x, hottest_y, hottest_z = max_idx

print(f"🔥 Hottest fluid found at X:{hottest_x}, Y:{hottest_y}, Z:{hottest_z}")
print(f"Slicing Top-Down exactly at Z-Height: {hottest_z}")

# 3. Slice horizontally at the hottest Z-plane
temp_slice = temp_3d[:, :, hottest_z]
mask_slice = mask[:, :, hottest_z]

# Mask out the solid metal
masked_slice = np.ma.masked_where(mask_slice == 0.0, temp_slice)

# 4. Create the heatmap
plt.figure(figsize=(8, 8))

# Plot the slice
plt.imshow(masked_slice.T, cmap='inferno', origin='lower')
plt.colorbar(label='Temperature (°C)')
plt.title(f"AI Digital Twin (Top-Down at Z={hottest_z})\nPower: {TEST_POWER}W | Velocity: {TEST_VELOCITY} m/s")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.tight_layout()
plt.show()

# --- 7. EXPORT TO PARAVIEW (FULL 3D) ---
print("\nExporting to 3D VTK format for ParaView...")
import pyvista as pv

# Create a 3D structured grid
grid = pv.ImageData()
grid.dimensions = np.array([128, 128, 128])

# VTK expects flattened 1D arrays (using Fortran 'F' ordering)
grid.point_data["Temperature"] = temp_3d.flatten(order="F")
grid.point_data["Pressure"] = real_physics[1].flatten(order="F")
grid.point_data["TKE"] = real_physics[2].flatten(order="F")

# Stack Vx, Vy, Vz into a single 3D vector field for streamlines
velocity_vectors = np.stack([
    real_physics[3], 
    real_physics[4], 
    real_physics[5]
], axis=-1)
grid.point_data["Velocity"] = velocity_vectors.reshape(-1, 3, order="F")

# Also export the geometry mask so you can visualize the solid walls!
grid.point_data["Solid_Geometry"] = mask.flatten(order="F")

# Save the file
out_name = f"Twin_Prediction_{TEST_POWER}W_{TEST_VELOCITY}mps.vti"
grid.save(out_name)
print(f"✅ Saved full 3D prediction to {out_name}!")

print("\nGenerating Professional 2D Matplotlib Slice...")

# 1. Calculate Total Velocity Magnitude
vx = real_physics[3]
vy = real_physics[4]
vz = real_physics[5]
vel_mag = np.sqrt(vx**2 + vy**2 + vz**2)

# 2. Extract the exact mid-plane slice (Snapping to dead-center Z=64)
# This will catch the core of the jet AND the exit vents!
Z_CENTER = 62 
slice_2d = vel_mag[:, :, Z_CENTER] 

# Rotate to match Ansys orientation
slice_2d = np.rot90(slice_2d)

# 3. Build the Presentation-Ready Plot
plt.figure(figsize=(10, 5))

# Using 'bicubic' to smooth the voxels, and 'auto' aspect to match your reference
plt.imshow(slice_2d, cmap='jet', origin='upper', 
           extent=[0.00, 0.10, 0.00, 0.027], 
           interpolation='bicubic', aspect='auto')

# 4. Format to exactly match your reference images
cbar = plt.colorbar(label='Velocity Magnitude (m/s)')
cbar.ax.tick_params(labelsize=10)

plt.xlabel('X-axis (Flow Axis) [m]', fontsize=12)
plt.ylabel('Y-axis (Height) [m]', fontsize=12)
plt.title('AI Prediction: 2D Mid-Plane Slice of Jet Impingement', fontsize=14)

plt.tight_layout()
plt.show()

# --- Extracting Quantitative Engineering Metrics ---
print("\n--- Engineering Metrics ---")

# The Temperature channel is index 0 in your real_physics array
temp_field = real_physics[0]

# Find the absolute hottest voxel in the entire 3D domain (The Chip Base)
max_chip_temp = np.max(temp_field)

# Calculate the Temperature Rise (Delta T) assuming 20C ambient inlet
ambient_temp = 20.0
delta_t = max_chip_temp - ambient_temp

print(f"Predicted Max Chip Temperature: {max_chip_temp:.2f} °C")
print(f"Predicted Temperature Rise (ΔT): {delta_t:.2f} °C")
print(f"Estimated Thermal Resistance: {(delta_t / TEST_POWER):.4f} °C/W")