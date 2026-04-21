"""
=============================================================================
  JET IMPINGEMENT DIGITAL TWIN — 3D VISUALIZER
  Shows Temperature, Pressure, and Velocity separately
=============================================================================
"""

import pandas as pd
import numpy as np
import plotly.express as px

print("Loading CFD Data...")
df = pd.read_csv("jet_prediction_results.csv")

# Calculate Velocity Magnitude
print("Calculating Velocity Magnitude...")
df['Velocity_Magnitude'] = np.sqrt(df['U_vel']**2 + df['V_vel']**2 + df['W_vel']**2)

# Downsample for smooth browser performance (25k points max)
if len(df) > 25000:
    df_plot = df.sample(n=25000, random_state=42)
else:
    df_plot = df

print(f"Rendering 3D visualizations with {len(df_plot):,} points...")

# Common layout settings
common_layout = dict(
    scene_camera=dict(eye=dict(x=1.8, y=1.8, z=0.8)),
    margin=dict(l=0, r=0, b=0, t=50),
    scene=dict(
        xaxis_title='X (m)',
        yaxis_title='Z (m)',
        zaxis_title='Y (m)',
        aspectmode='cube'
    )
)

# ==================== 1. TEMPERATURE ====================
fig_temp = px.scatter_3d(
    df_plot,
    x='X', y='Z', z='Y',
    color='Temperature_C',
    color_continuous_scale='RdYlBu_r',      # Hot = red, cold = blue
    title='Digital Twin — Temperature Field (°C)',
    labels={'Temperature_C': 'Temperature (°C)'}
)
fig_temp.update_traces(marker=dict(size=2.8, opacity=0.85))
fig_temp.update_layout(**common_layout)
fig_temp.show()

# ==================== 2. PRESSURE ====================
fig_press = px.scatter_3d(
    df_plot,
    x='X', y='Z', z='Y',
    color='Pressure_Pa',
    color_continuous_scale='Viridis',        # Good for pressure gradients
    title='Digital Twin — Pressure Field (Pa)',
    labels={'Pressure_Pa': 'Pressure (Pa)'}
)
fig_press.update_traces(marker=dict(size=2.8, opacity=0.85))
fig_press.update_layout(**common_layout)
fig_press.show()

# ==================== 3. VELOCITY MAGNITUDE ====================
fig_vel = px.scatter_3d(
    df_plot,
    x='X', y='Z', z='Y',
    color='Velocity_Magnitude',
    color_continuous_scale='Turbo',          # Classic fluid velocity scale
    title='Digital Twin — Velocity Magnitude (m/s)',
    labels={'Velocity_Magnitude': 'Speed (m/s)'}
)
fig_vel.update_traces(marker=dict(size=2.8, opacity=0.85))
fig_vel.update_layout(**common_layout)
fig_vel.show()

print("✅ All three 3D visualizations opened in browser!")
print("   • Temperature (RdYlBu_r)")
print("   • Pressure (Viridis)")
print("   • Velocity Magnitude (Turbo)")