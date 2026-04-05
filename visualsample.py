import os
import glob
import torch
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

# THE FIX: Import the dataset class so PyTorch knows how to unpack the .pt file
from FeatureEngineering import JetImpingementDataset

DATA_DIR = r"D:\data\JET"

print("Discovering training simulation files...")
train_sim_dir = os.path.join(DATA_DIR, 'train_sims')
train_sim_files = sorted(glob.glob(os.path.join(train_sim_dir, '*.pt')))

if not train_sim_files:
    raise FileNotFoundError(f"No simulation files found in {train_sim_dir}. Did you run FeatureEngineering.py?")

# Grab the very first simulation file
first_sim_path = train_sim_files[0]
print(f"Loading simulation dataset: {os.path.basename(first_sim_path)}")

sim_ds = torch.load(first_sim_path, weights_only=False)

# Extract tensors and remove the batch dimension (using .X and .T directly)
X_sim = sim_ds.X.squeeze(0).numpy()
T_sim = sim_ds.T.squeeze(0).numpy()

print("Loading scalers...")
scaler_spatial = joblib.load(os.path.join(DATA_DIR, 'scaler_spatial.pkl'))
scaler_vel = joblib.load(os.path.join(DATA_DIR, 'scaler_vel.pkl'))

# In your FeatureEngineering pipeline, indices 4 through 9 are the Spatial columns
spatial_scaled = X_sim[:, 4:10]

print("Unscaling coordinates and fluid properties...")
spatial_real = scaler_spatial.inverse_transform(spatial_scaled)

# Extract real X, Y, Z
xs, ys, zs = spatial_real[:, 0], spatial_real[:, 1], spatial_real[:, 2]

# Extract and unscale velocity to color the point cloud
uvw_scaled = T_sim[:, 2:5]
uvw_real = scaler_vel.inverse_transform(uvw_scaled)

# Calculate absolute fluid speed (m/s)
speed = np.sqrt(uvw_real[:, 0]**2 + uvw_real[:, 1]**2 + uvw_real[:, 2]**2)

# Create a DataFrame for plotting
df = pd.DataFrame({
    'X': xs,
    'Y': ys,
    'Z': zs,
    'Speed': speed
})

# Plotly handles ~30k points buttery smooth in the browser. 
df_plot = df.sample(n=30000, random_state=42)

print("Rendering 3D Sampling Distribution in your browser...")

# Create the 3D scatter plot
fig = px.scatter_3d(
    df_plot, 
    x='X', 
    y='Z',   # Swapped Y and Z to lay the impingement wall flat
    z='Y', 
    color='Speed', # Color by Speed to visually prove the sampling worked
    color_continuous_scale='Turbo',
    title='Adaptive Mesh Verification: Probability-Weighted Sampling',
    labels={'Speed': 'Velocity (m/s)', 'Y': 'Height (m)'}
)

# Make points small and semi-transparent to see the density clusters
fig.update_traces(marker=dict(size=2.0, opacity=0.6))

# Set the camera to look at the full channel domain
fig.update_layout(
    scene_camera=dict(eye=dict(x=0.0, y=-2.0, z=0.5)),
    margin=dict(l=0, r=0, b=0, t=40)
)

fig.show()