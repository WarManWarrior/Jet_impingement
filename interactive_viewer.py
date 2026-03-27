# import pyvista as pv
# import numpy as np
# import os

# # --- Configuration ---
# vti_file = "Twin_Prediction_62.0W_-7.5mps.vti"

# if not os.path.exists(vti_file):
#     print(f"❌ Cannot find {vti_file}. Make sure you ran test_twin.py first!")
#     exit()

# print(f"Loading {vti_file} into 3D Volumetric Flow Viewer...")
# mesh = pv.read(vti_file)

# # 1. Calculate Velocity Magnitude
# if "Velocity" in mesh.point_data:
#     vel_vectors = mesh.point_data["Velocity"]
#     mesh.point_data["Velocity_Magnitude"] = np.linalg.norm(vel_vectors, axis=1)

# # --- 2. Build the 3D Plotter ---
# plotter = pv.Plotter(window_size=[1200, 800])

# # THE MAGIC: Volume Rendering
# # 'opacity="linear"' is the secret sauce. It automatically makes 0 m/s completely transparent 
# # and smoothly fades the colors in as the velocity increases. No more jagged blocks!
# plotter.add_volume(mesh, 
#                    scalars="Velocity_Magnitude", 
#                    cmap="jet",          # Classic CFD colors
#                    opacity="linear",    # Fades out the dead air seamlessly
#                    mapper="smart")      # Uses your GPU to render it like realistic smoke

# # Add a faint wireframe box so you can see the chamber walls
# plotter.add_mesh(mesh.outline(), color="white", line_width=2)

# plotter.add_text("Volumetric Render: Continuous Flow Plume", font_size=14)
# plotter.add_axes()

# print("✅ Volumetric Viewer launched! Click and drag to rotate.")
# plotter.show()
import pyvista as pv
import numpy as np
import os

# --- Configuration ---
vti_file = "Twin_Prediction_62.0W_-7.5mps.vti"

if not os.path.exists(vti_file):
    print(f"❌ Cannot find {vti_file}. Make sure you ran test_twin.py first!")
    exit()

print(f"Loading {vti_file} into Flow Pattern Viewer...")
mesh = pv.read(vti_file)

# 1. Activate the 3D Vectors
mesh.set_active_vectors("Velocity")
vel_mag = np.linalg.norm(mesh.point_data["Velocity"], axis=1)
mesh.point_data["Velocity_Magnitude"] = vel_mag

print("Tracing virtual particle streamlines...")

# 2. The Math Trace (Calculated on CPU)
streamlines = mesh.streamlines(
    vectors="Velocity",
    source_center=(64, 64, 62),  
    source_radius=15.0,          # Catch the whole nozzle
    n_points=800,                # We can safely bump this up now!
    integration_direction="both",
    max_steps=2500               
)

# 3. Build the 3D Plotter
plotter = pv.Plotter(window_size=[1200, 800])

if streamlines.n_points == 0:
    print("⚠️ No flow caught! The particles missed the jet.")
else:
    print("Sending to GPU for hardware-accelerated rendering...")
    
    # 🔥 THE GPU FIX: Notice we deleted .tube() entirely!
    # We pass the raw streamlines and let the GPU shader draw the tubes instantly.
    plotter.add_mesh(streamlines, 
                     scalars="Velocity_Magnitude", 
                     cmap="jet", 
                     render_lines_as_tubes=True,  # <-- GPU Hardware Acceleration!
                     line_width=4,                # GPU Tube thickness
                     show_scalar_bar=True,
                     scalar_bar_args={"title": "Flow Direction & Speed (m/s)"})

# Add the faint wireframe box
plotter.add_mesh(mesh.outline(), color="white", line_width=2)

plotter.add_text("GPU-Accelerated 3D Particle Streamlines", font_size=14)
plotter.add_axes()

print("✅ Flow Pattern Viewer launched!")
plotter.show()