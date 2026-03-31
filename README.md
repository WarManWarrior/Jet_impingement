# Jet Impingement CFD: Fourier Neural Operator (FNO) & Graph Neural Network (GNN)

This repository contains a state-of-the-art research pipeline for building **AI-based thermal surrogates** for jet impingement cooling. The system is designed to replace million-node CFD simulations with lightning-fast neural operators (3D FNO) and graph-based models (GNN).

---

## 🌪️ Project Overview
Jet impingement is a critical cooling technology for electronics and power systems. CFD simulations are computationally expensive (~12 million nodes per run). This project implements two main architectures to predict 3D Temperature (T), Pressure (P), and Velocity (V) fields:

1.  **3D Fourier Neural Operator (FNO)**: Processes structured $64^3$ voxel grids for extreme global spatial capture.
2.  **Edge-Conditioned GNN**: Processes unstructured 20k-node reduced meshes for localized geometric precision.

---

## 📂 Dataset Specification
*   **Source Data**: Located in `D:\data\JET\H4` and `D:\data\JET\H5`.
*   **Geometric Variants**:
    *   **H4**: Nozzle diameter $D = H/4$.
    *   **H5**: Nozzle diameter $D = H/5$.
*   **Simulation Count**: 166 independent simulations across varying inlet Velocities (3-12 m/s) and Input Power (40-85 W).
*   **Raw Resolution**: Each HDF5 file contains ~11.9 million nodes.

---

## 🛠️ File Manifest & Pipeline Flow

### 1. Raw CFD Diagnostics (Pre-Normalization)
*   **`raw_data_eda.py`**: 
    *   **Purpose**: Scans all 166 HDF5 files to perform a physical and geometric audit.
    *   **Insights**: Verifies Temperature is in Celsius [20-62°C], checks 100% geometric alignment of bounding boxes, and confirms node counts (~12M/mesh).
    *   **Outputs**: `/raw_eda_results/` (Histograms, Bounding box alignment).

### 2. Structured FNO Pipeline
*   **`fno_prep.py`**: 
    *   **Purpose**: Voxelizes the 12M nodes into a $64 \times 64 \times 64$ structured grid.
    *   **Key Logic**: Geometric prefixing (`H4_`/`H5_`) to prevent filename collisions. Velocity-weighted voxelization to preserve jet intensity.
    *   **Data Saving**: Organizes processed tensors into `D:\data\JET\FNO_Prepared`.
*   **`pca_analysis.py`**: 
    *   **Purpose**: Performs Principal Component Analysis on the 64^3 grids.
    *   **Insights**: Identifies the 10 dominant flow/thermal modes. Correlates PC scores with physical parameters (Velocity, Power, Diameter).
    *   **Outputs**: `/pca_results/` (Explained variance, Spatial mode slices).
*   **`comprehensive_eda.py`**: 
    *   **Purpose**: Post-voxelization statistical scan for data integrity and spatial profiles (centerline T-profile).
    *   **Outputs**: `/eda_results/`.

### 3. Unstructured GNN Pipeline (Baseline)
*   **`preprocess.py`**: 
    *   **Purpose**: Implements physics-aware data reduction using "Importance Scoring" (Variance + Gradient).
    *   **Logic**: Uses K-Means to cluster 12M nodes down to 20k structural "anchor nodes."
*   **`gnn_pipeline.py` & `digital_twin.py`**: 
    *   **Purpose**: Training logic and model architecture using `NNConv` and complex edge features `[dist, dx, dy, dz]`.
*   **`interactive_viewer.py`**: 
    *   **Purpose**: Real-time 3D Plotly rendering for GNN results.

---

## 🚀 Execution Instructions

### A. FNO Diagnostic & Preprocessing Flow
1.  **Direct CFD Audit**:
    `python raw_data_eda.py`
2.  **Generate Structured Grids**:
    `python fno_prep.py`
3.  **Physical Parametric Analysis**:
    `python pca_analysis.py`
    `python comprehensive_eda.py`

### B. GNN Reduction Flow
1.  **Reduce Mesh Density**:
    `python preprocess.py`
2.  **Scale and Cluster**:
    `python train_twin.py`

---

## 📊 Key Results to Date
*   **Data Reduction**: Achieved **45x compression** from raw 11.9M nodes to a structures 64^3 grid.
*   **Consistency**: Verified 100% geometric synchronization across H4 and H5 files.
*   **Normalization**: Implemented global Welford-based 2-pass normalization to ensure stable cross-dataset training.

---

## ⚙️ Core Dependencies
*   `torch` (GPU Acceleration recommended)
*   `h5py` (HDF5 data access)
*   `numpy` & `scipy`
*   `matplotlib` & `seaborn` (Visualizations)
*   `sklearn` (PCA & Clustering)
