import pandas as pd
import numpy as np
import plotly.express as px

print("Loading CFD Data...")
df = pd.read_csv("jet_prediction_results.csv")

# 1. Calculate Velocity Magnitude (Speed) from the U, V, W vectors
print("Calculating Velocity Magnitude...")
df['Velocity_Magnitude'] = np.sqrt(df['U_vel']**2 + df['V_vel']**2 + df['W_vel']**2)

# 2. Downsample for smooth browser performance (25k points)
if len(df) > 25000:
    df_plot = df.sample(n=25000, random_state=42)
else:
    df_plot = df

print("Rendering 3D Browser Visualization...")

# 3. Create the 3D interactive plot
fig = px.scatter_3d(
    df_plot, 
    x='X', 
    y='Z',  # Swapped Y and Z to lay the impingement wall flat
    z='Y', 
    color='Velocity_Magnitude', 
    color_continuous_scale='Turbo',  # 'Turbo' is great for fluid velocity
    title='Digital Twin: Jet Impingement Velocity Field (m/s)',
    labels={'Velocity_Magnitude': 'Speed (m/s)'}
)

# Make the points smaller and slightly transparent
fig.update_traces(marker=dict(size=2.5, opacity=0.8))

# Tweak the camera angle to look at the stagnation zone
fig.update_layout(
    scene_camera=dict(eye=dict(x=1.5, y=1.5, z=0.5)),
    margin=dict(l=0, r=0, b=0, t=40)
)

# Open in browser
fig.show()

print("✓ Velocity visualizer opened in browser!")