# Jet Impingement CFD Graph Neural Surrogate

This repository contains a complete, physics-aware Graph Neural Network (GNN) pipeline designed to act as a **lightning-fast surrogate model** for fluid dynamics and heat transfer simulations (CFD) in Jet Impingement cooling. 

By replacing expensive Navier-Stokes solvers with an edge-conditioned Neural Operator, this system ingests boundary condition parameters (Velocity, Power) and natively renders high-fidelity 3D spatial Temperature and Pressure distributions in milliseconds.

---

## 🚀 Pipeline Architecture

The system is broken down into three core operational scripts that must be executed sequentially: data reduction, neural training, and deployment evaluation.

### 1. `preprocess.py` (Data Engineering & Spatial Reduction)
CFD meshes natively produce millions of nodes (~1.7M+), which causes Out-Of-Memory (OOM) failures for Graph convolutions. The preprocessing script intelligently scales down the physics.

* **Feature Engineering:** Extracts spatial velocity gradients, Turbulent Kinetic Energy (TKE), and Temperature deviations to assign physical "importance scores" to every node.
* **Smart Filtering:** Removes boundary masking and low-activity stagnant zones.
* **Clustering Extraction:** Deploys an optimized K-Means structural reduction to physically cluster the 1.7M fluid domain down into a lightweight `Training_Clustered_20K` mesh for training constraints and a denser `Validation_Masked_238K` mesh for rendering.
* **Output:** Saves isolated `.h5` files optimized structurally for PyTorch Geometric (PyG).

### 2. `gnn_pipeline.py` (Physics-Aware Core Surrogate)
The heart of the Neural Operator. This script trains the model strictly avoiding Data Leakage (it hides CFD flow arrays and learns exclusively from input physical bounds).

* **Boundary Constraints Check:** Parses dataset file names to extract global `[Velocity, Power]` domains and enforce raw min-max normalizations mapping to scale variables natively `(vel_in - 3)/7` and `(pow_in - 40)/40`.
* **Graph Convolution (`ThermalGNN`)**: Engineers a 7-channel dynamic topology wrapped mapping spatial edge relationships `[distance, dx, dy, dz]` cleanly into PyTorch Geometric's `NNConv` mechanism. 
* **Hardware Optimizations**: Runs completely via PyTorch's Automatic Mixed Precision (`torch.amp`) restricting convolutions to `k=6` neighbors and `hidden=32` filters bounding GPU footprints securely.
* **Physics Loss Framework**: Maps standard Data Loss alongside a continuous spatial mapping Laplacian (`smoothness_loss`) restricting isolated thermodynamic node fracturing dynamically.
* **Output:** Writes the absolute scalar mappings natively alongside the neural mapping parameters locally into `thermal_gnn.pth`.

### 3. `evaluate_surrogate.py` (Quantitative Visual Analytics)
Executes runtime inference and model diagnostics testing using fully unseen boundary domains.

* **State Reloading**: Autoloads explicit `thermal_gnn.pth` network architectures and dimensional weights cleanly.
* **Interpolation Deployment**: Computes arbitrary, unseen interpolation logic (i.e. `Velocity: 6.3m/s, Power: 52W`) natively bridging thermodynamic outputs statically across PyTorch grids.
* **Quantitative Analysis**: Generates Root Mean Squared Error (`RMSE`) and Mean Absolute Error (`MAE`) against real baseline structural coordinates natively.
* **3D Matplotlib Rendering**: Dynamically sweeps and exports rich mathematical visualizations locally into the `/visualizations` directory, mapping gradients directly to target thermal profiles `[target °C]`.

---

## ⚡ Execution Instructions

**Step 1: Parse the Raw CFD Sim Volumes**
Ensure your mesh sets exist at root paths.
```bash
python preprocess.py
```

**Step 2: Train the Neural Operator**
Adjust the strict drive endpoints (e.g. `D:\data\JET\...`) directly pointing to the exported `Training_Clustered_20K` directories.
```bash
python gnn_pipeline.py
```
*(Model topology and standardization states will automatically compile tightly into `thermal_gnn.pth`.)*

**Step 3: Validate Predictions & Map Heatmaps**
Once the checkpoint exists locally, trigger testing evaluations extracting target 3D spatial interpolations.
```bash
python evaluate_surrogate.py
```
Check your `/visualizations` path for direct visual evaluation validation graphics comparing to your base CFD mesh.

---

## ⚙️ Dependencies

* `torch` (Hardware explicit CUDA enabled variant highly recommended for NNConv execution)
* `torch_geometric`
* `h5py` for binary structure arrays
* `numpy`
* `matplotlib` for 3D topological Heatmap renderings
