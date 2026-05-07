"""
=============================================================================
  JET IMPINGEMENT — INTERACTIVE HEATMAP DASHBOARD
  Reads jet_prediction_results.csv and opens a multi-panel dashboard in the
  browser. Saves a standalone jet_dashboard.html you can share or reopen.

  Run: python visualize.py

  Views:
    1. Chip Surface Heatmap  — Temperature on the chip wall (y ≈ 0)
    2. XY Cross-Section      — Field slice at Z = jet_center_z
    3. XZ Top-Down Slice     — Field slice at a chosen Y level
    4. 3D Scatter            — Downsampled full-domain point cloud
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ──────────────────────────────────────────────────────────────────
CSV_PATH    = "jet_prediction_results.csv"
OUTPUT_HTML = "jet_dashboard.html"

# Physical geometry (must match your simulation)
JET_CX = 0.05238   # jet center X (m)
JET_CZ = 0.012     # jet center Z (m)

# Color scales: each field gets a purpose-built scale
CSCALE = {
    "Temperature_C"      : "Turbo",
    "Pressure_Pa"        : "RdBu_r",
    "Velocity_Magnitude" : "Plasma",
    "V_vel"              : "RdBu_r",
}
FIELD_LABEL = {
    "Temperature_C"      : "Temperature (°C)",
    "Pressure_Pa"        : "Pressure (Pa)",
    "Velocity_Magnitude" : "Speed (m/s)",
    "V_vel"              : "Vy — Vertical Velocity (m/s)",
}

# ─── LOAD ─────────────────────────────────────────────────────────────────────
print("Loading CSV …")
df = pd.read_csv(CSV_PATH)
print(f"  {len(df):,} rows  |  columns: {list(df.columns)}")

# Round coordinates to snap to mesh layers
df["X_r"] = np.round(df["X"], 6)
df["Y_r"] = np.round(df["Y"], 6)
df["Z_r"] = np.round(df["Z"], 6)

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def percentile_clim(arr, lo=1, hi=99):
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


def to_regular_grid(px, pz, pval, nx=300, nz=200):
    """Scatter → regular grid via linear interpolation."""
    xi = np.linspace(px.min(), px.max(), nx)
    zi = np.linspace(pz.min(), pz.max(), nz)
    Xi, Zi = np.meshgrid(xi, zi)
    Vi = griddata((px, pz), pval, (Xi, Zi), method="linear")
    return xi * 1e3, zi * 1e3, Vi   # return in mm


def to_regular_grid_xy(px, py, pval, nx=300, ny=150):
    xi = np.linspace(px.min(), px.max(), nx)
    yi = np.linspace(py.min(), py.max(), ny)
    Xi, Yi = np.meshgrid(xi, yi)
    Vi = griddata((px, py), pval, (Xi, Yi), method="linear")
    return xi * 1e3, yi * 1e3, Vi


# ─── PANEL 1: CHIP SURFACE HEATMAP ───────────────────────────────────────────
print("Building chip surface heatmaps …")
wall = df[df["Y_r"] < 5e-4].copy()   # y < 0.5 mm

surface_figs = {}
for field in ["Temperature_C", "Pressure_Pa", "Velocity_Magnitude"]:
    xi, zi, Vi = to_regular_grid(wall["X_r"].values, wall["Z_r"].values,
                                  wall[field].values)
    vlo, vhi = percentile_clim(wall[field].values)
    surface_figs[field] = dict(xi=xi, zi=zi, Vi=Vi, vlo=vlo, vhi=vhi)


# ─── PANEL 2: XY CROSS-SECTION AT Z ≈ JET_CZ ─────────────────────────────────
print("Building XY cross-section …")
yz_levels = np.unique(df["Z_r"].values)
z_target  = yz_levels[np.argmin(np.abs(yz_levels - JET_CZ))]
xysec = df[np.abs(df["Z_r"] - z_target) < 5e-4].copy()
print(f"  Z cross-section at Z={z_target*1e3:.2f} mm  ({len(xysec):,} nodes)")

xy_figs = {}
for field in ["Temperature_C", "Pressure_Pa", "V_vel", "Velocity_Magnitude"]:
    xi, yi, Vi = to_regular_grid_xy(xysec["X_r"].values, xysec["Y_r"].values,
                                     xysec[field].values)
    vlo, vhi = percentile_clim(xysec[field].values)
    xy_figs[field] = dict(xi=xi, yi=yi, Vi=Vi, vlo=vlo, vhi=vhi)


# ─── PANEL 3: XZ TOP-DOWN SLICES ─────────────────────────────────────────────
print("Building XZ horizontal slices …")
y_levels = np.sort(np.unique(df["Y_r"].values))
# Pick ~8 representative Y levels spanning the domain
y_picks  = np.interp(np.linspace(0, 1, 8), np.linspace(0, 1, len(y_levels)), y_levels)
y_picks  = [y_levels[np.argmin(np.abs(y_levels - yp))] for yp in y_picks]
print(f"  Y slices (mm): {[f'{yp*1e3:.1f}' for yp in y_picks]}")

xz_slices = {}
for yp in y_picks:
    sl = df[np.abs(df["Y_r"] - yp) < 2e-4]
    if len(sl) < 50:
        continue
    xi, zi, Vi = to_regular_grid(sl["X_r"].values, sl["Z_r"].values,
                                  sl["Temperature_C"].values)
    xz_slices[yp] = dict(xi=xi, zi=zi, Vi=Vi, label=f"Y = {yp*1e3:.2f} mm")


# ─── PANEL 4: 3D SCATTER (DOWNSAMPLED) ───────────────────────────────────────
print("Sampling 3D scatter …")
rng = np.random.default_rng(42)
ds_idx = rng.choice(len(df), size=min(20_000, len(df)), replace=False)
ds = df.iloc[ds_idx]


# ─── BUILD PLOTLY FIGURE ──────────────────────────────────────────────────────
print("Assembling Plotly dashboard …")

# ── Subplot layout ────────────────────────────────────────────────────────────
figs = []

def hm(xi, zi, Vi, cs, vlo, vhi, xlabel="X (mm)", ylabel="Z (mm)"):
    """Return a Heatmap trace."""
    return go.Heatmap(
        x=xi, y=zi, z=Vi,
        colorscale=cs, zmin=vlo, zmax=vhi,
        colorbar=dict(thickness=12, tickfont=dict(size=9)),
        showscale=True,
    )

axis_kw = dict(showgrid=True, gridcolor="#e1e4e8", linecolor="#e1e4e8", zeroline=False, showticklabels=True, tickfont=dict(size=8))
layout_kw = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(color="#000000", family="'Courier New', monospace"),
    margin=dict(l=60, r=100, t=60, b=60),
)

# Chip surface
for field in ["Temperature_C", "Pressure_Pa", "Velocity_Magnitude"]:
    f = go.Figure()
    d = surface_figs[field]
    t = hm(d["xi"], d["zi"], d["Vi"], CSCALE[field], d["vlo"], d["vhi"])
    t.colorbar.title = dict(text=FIELD_LABEL[field], side="right")
    f.add_trace(t)
    f.add_trace(go.Scatter(
        x=[JET_CX * 1e3], y=[JET_CZ * 1e3],
        mode="markers", marker=dict(symbol="cross", size=10, color="black", line=dict(width=1.5, color="white")),
        showlegend=False, name="Jet axis"
    ))
    f.update_layout(title=dict(text=f"Chip Surface — {FIELD_LABEL[field]}", x=0.5, font=dict(size=16, color="#000000")), **layout_kw)
    f.update_xaxes(title_text="X (mm)", **axis_kw)
    f.update_yaxes(title_text="Z (mm)", **axis_kw)
    figs.append(f)

# XY cross-sections
for field in ["Temperature_C", "V_vel", "Velocity_Magnitude"]:
    f = go.Figure()
    d = xy_figs[field]
    t = hm(d["xi"], d["yi"], d["Vi"], CSCALE[field], d["vlo"], d["vhi"])
    t.colorbar.title = dict(text=FIELD_LABEL[field], side="right")
    f.add_trace(t)
    f.update_layout(title=dict(text=f"XY Section (Z=jet center) — {FIELD_LABEL[field]}", x=0.5, font=dict(size=16, color="#000000")), **layout_kw)
    f.update_xaxes(title_text="X (mm)", **axis_kw)
    f.update_yaxes(title_text="Y (mm)", **axis_kw)
    figs.append(f)

# XZ Y-level slices (two representative ones)
y_keys = list(xz_slices.keys())
for yp in y_keys[:2]:
    f = go.Figure()
    d = xz_slices[yp]
    vlo, vhi = percentile_clim(d["Vi"][~np.isnan(d["Vi"])].flatten())
    t = hm(d["xi"], d["zi"], d["Vi"], "Turbo", vlo, vhi)
    t.colorbar.title = dict(text="Temperature (°C)", side="right")
    f.add_trace(t)
    f.update_layout(title=dict(text=f"XZ Slice (Y levels) — Temperature | {d['label']}", x=0.5, font=dict(size=16, color="#000000")), **layout_kw)
    f.update_xaxes(title_text="X (mm)", **axis_kw)
    f.update_yaxes(title_text="Z (mm)", **axis_kw)
    figs.append(f)

# 3D scatter
f = go.Figure()
f.add_trace(go.Scatter3d(
    x=ds["X"] * 1e3, y=ds["Z"] * 1e3, z=ds["Y"] * 1e3,
    mode="markers",
    marker=dict(
        size=1.5,
        color=ds["Temperature_C"],
        colorscale="Turbo",
        opacity=0.7,
        colorbar=dict(thickness=12, title=dict(text="Temperature (°C)", side="right"), tickfont=dict(size=9)),
    ),
    showlegend=False,
))
f.update_layout(title=dict(text="3D Point Cloud (20k pts) — Temperature", x=0.5, font=dict(size=16, color="#000000")), **layout_kw)
f.update_scenes(xaxis_title="X (mm)", yaxis_title="Z (mm)", zaxis_title="Y (mm)")
figs.append(f)

# ─── ADD INTERACTIVE DROPDOWN SLICES ──────────────────────────────────────────
# Build dropdown to swap between XZ Y-level slices (attached to row 3, col 1&2)
# This is encoded as updatemenus buttons that swap z data
print("Building dropdown for Y-slice selection …")

# We'll create a separate standalone figure for the Y-slice explorer
fig_slices = go.Figure()

for yp, d in xz_slices.items():
    vlo, vhi = percentile_clim(d["Vi"][~np.isnan(d["Vi"])].flatten())
    fig_slices.add_trace(go.Heatmap(
        x=d["xi"], y=d["zi"], z=d["Vi"],
        colorscale="Turbo", zmin=vlo, zmax=vhi,
        colorbar=dict(title="Temperature (°C)"),
        name=d["label"],
        visible=False,
    ))

fig_slices.data[0].visible = True

buttons = [
    dict(
        label=xz_slices[yp]["label"],
        method="update",
        args=[{"visible": [i == idx for i in range(len(xz_slices))]},
              {"title": f"XZ Temperature Slice — {xz_slices[yp]['label']}"}],
    )
    for idx, yp in enumerate(xz_slices.keys())
]

fig_slices.update_layout(
    title=dict(text="XZ Temperature Slice — " + list(xz_slices.values())[0]["label"],
               font=dict(size=16, family="'Courier New', monospace", color="#000000"), x=0.5),
    updatemenus=[dict(
        buttons=buttons,
        direction="down",
        showactive=True,
        x=0.02, xanchor="left",
        y=1.12, yanchor="top",
        bgcolor="#ffffff",
        bordercolor="#000000",
        font=dict(color="#000000", size=12),
    )],
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(color="#000000", family="'Courier New', monospace"),
    xaxis_title="X (mm)",
    yaxis_title="Z (mm)",
    margin=dict(l=60, r=80, t=100, b=60),
)
fig_slices.update_xaxes(gridcolor="#e1e4e8", linecolor="#e1e4e8")
fig_slices.update_yaxes(gridcolor="#e1e4e8", linecolor="#e1e4e8")


# ─── ADD FIELD COMPARISON FIGURE ──────────────────────────────────────────────
print("Building field selector for chip surface …")

fig_surface = go.Figure()
fields_surface = ["Temperature_C", "Pressure_Pa", "Velocity_Magnitude"]
for i, field in enumerate(fields_surface):
    d = surface_figs[field]
    fig_surface.add_trace(go.Heatmap(
        x=d["xi"], y=d["zi"], z=d["Vi"],
        colorscale=CSCALE[field], zmin=d["vlo"], zmax=d["vhi"],
        colorbar=dict(title=FIELD_LABEL[field]),
        name=FIELD_LABEL[field],
        visible=(i == 0),
    ))

surface_buttons = [
    dict(
        label=FIELD_LABEL[f],
        method="update",
        args=[{"visible": [j == i for j in range(len(fields_surface))]},
              {"title": f"Chip Surface — {FIELD_LABEL[f]}"}],
    )
    for i, f in enumerate(fields_surface)
]

# Add stagnation point marker
fig_surface.add_trace(go.Scatter(
    x=[JET_CX * 1e3], y=[JET_CZ * 1e3],
    mode="markers+text",
    marker=dict(symbol="cross", size=12, color="black", line=dict(width=2)),
    text=["Jet axis"], textposition="top right",
    textfont=dict(color="black", size=10),
    showlegend=False,
))

fig_surface.update_layout(
    title=dict(text="Chip Surface — Temperature", x=0.5,
               font=dict(size=16, family="'Courier New', monospace", color="#000000")),
    updatemenus=[dict(
        buttons=surface_buttons,
        direction="down",
        showactive=True,
        x=0.02, xanchor="left",
        y=1.12, yanchor="top",
        bgcolor="#ffffff",
        bordercolor="#000000",
        font=dict(color="#000000", size=12),
    )],
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(color="#000000", family="'Courier New', monospace"),
    xaxis_title="X (mm)",
    yaxis_title="Z (mm)",
    margin=dict(l=60, r=80, t=100, b=60),
)
fig_surface.update_xaxes(gridcolor="#e1e4e8", linecolor="#e1e4e8")
fig_surface.update_yaxes(gridcolor="#e1e4e8", linecolor="#e1e4e8")


# ─── WRITE COMBINED HTML ──────────────────────────────────────────────────────
print(f"Writing {OUTPUT_HTML} …")

from plotly.io import to_html

html_parts = [
    """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Jet Impingement — Heatmap Dashboard</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600&display=swap');
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #ffffff;
    color: #000000;
    font-family: 'Rajdhani', sans-serif;
    padding: 0 0 60px 0;
  }
  header {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-bottom: 1px solid #dee2e6;
    padding: 24px 48px 20px;
    display: flex;
    align-items: center;
    gap: 20px;
  }
  .logo {
    width: 40px; height: 40px;
    background: conic-gradient(from 180deg, #58a6ff, #1f6feb, #58a6ff);
    border-radius: 8px;
    flex-shrink: 0;
  }
  header h1 {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.4rem;
    color: #000000;
    letter-spacing: 0.05em;
  }
  header p {
    font-size: 0.85rem;
    color: #495057;
    margin-top: 2px;
  }
  .badge {
    margin-left: auto;
    background: #ffffff;
    border: 1px solid #ced4da;
    border-radius: 6px;
    padding: 6px 14px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #000000;
  }
  .section {
    margin: 36px 32px 0;
  }
  .section-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: #1f6feb;
    text-transform: uppercase;
    margin-bottom: 10px;
    padding-left: 4px;
    border-left: 3px solid #1f6feb;
    padding-left: 10px;
  }
  .card {
    background: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 12px;
    overflow: hidden;
    width: 70%;
    margin: 0 auto 30px auto;
    aspect-ratio: 1 / 1;
  }
  .grid2 {
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  footer {
    text-align: center;
    padding: 40px;
    font-size: 0.75rem;
    color: #6c757d;
    font-family: 'Share Tech Mono', monospace;
  }
</style>
</head>
<body>
<header>
  <div class="logo"></div>
  <div>
    <h1>JET IMPINGEMENT DIGITAL TWIN</h1>
    <p>CFD Surrogate — Field Heatmap Dashboard</p>
  </div>
  <div class="badge">250,000 NODES</div>
</header>

<div class="section">
  <div class="section-label">01 — Overview: All Fields × All Views</div>
""",
    "\n".join([f'<div class="card">{to_html(f, full_html=False, include_plotlyjs="cdn" if i==0 else False, default_width="100%", default_height="100%")}</div>' for i, f in enumerate(figs)]),
    """
</div>

<div class="section" style="margin-top:28px;">
  <div class="section-label">02 — Chip Surface Field Explorer</div>
  <div class="grid2">
    <div class="card">
""",
    to_html(fig_surface, full_html=False, include_plotlyjs=False, default_width="100%", default_height="100%"),
    """    </div>
    <div class="card">
""",
    to_html(fig_slices, full_html=False, include_plotlyjs=False, default_width="100%", default_height="100%"),
    """    </div>
  </div>
</div>

<footer>
  DIGITAL TWIN CFD SURROGATE &nbsp;·&nbsp; LATENT GNN &nbsp;·&nbsp;
  Data: jet_prediction_results.csv
</footer>
</body>
</html>
""",
]

with open(OUTPUT_HTML, "w", encoding="utf-8") as fh:
    fh.write("".join(html_parts))

print(f"\n✓  Dashboard saved → {OUTPUT_HTML}")
print("  Opening in browser …")

import webbrowser, os, pathlib
webbrowser.open(pathlib.Path(OUTPUT_HTML).resolve().as_uri())
