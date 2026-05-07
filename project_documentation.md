# Jet Impingement CFD Graph Neural Surrogate: Detailed Architecture & Design Decisions

This document provides a highly detailed, comprehensive breakdown of the methodology, engineering decisions, and architecture behind the Jet Impingement CFD Graph Neural Surrogate project.

---

## 1. Executive Summary

This project implements a physics-aware Graph Neural Network (GNN) pipeline designed to act as a **lightning-fast surrogate model** for fluid dynamics and heat transfer simulations (CFD) in Jet Impingement cooling. Traditional CFD solvers (solving Navier-Stokes equations) are computationally expensive and slow. This pipeline replaces the solver with an edge-conditioned Neural Operator. 

It takes boundary conditions (Velocity, Power, H/D ratio) as inputs and predicts high-fidelity 3D spatial Temperature, Pressure, and Velocity fields ($U, V, W$) in milliseconds.

---

## 2. The Core Problem: Data Scale vs. Memory Bottlenecks

A raw Jet Impingement CFD mesh naturally contains upwards of **1.7 million to 11.9 million nodes** depending on the specific geometry. Standard Graph Convolutional Networks (GCNs) require nodes to pass messages to their neighbors simultaneously in GPU memory. 

Attempting to load a graph with millions of nodes and edges into PyTorch Geometric (PyG) results in catastrophic **Out-Of-Memory (OOM) failures**, even on high-end hardware. Thus, the overarching engineering challenge of this project was balancing physical fidelity with hardware constraints.

The solution was a multi-tiered data reduction and model abstraction pipeline:
1. **Node Sub-sampling** (1.7M → 250k nodes)
2. **Latent Space Graph Mapping** (250k Fine nodes ↔ 3k Latent nodes)

---

## 3. Data Reduction & Feature Engineering (`FeatureEngineering.py`)

To resolve the node-count explosion, the raw `.h5` CFD simulation data undergoes intelligent spatial reduction down to **250,000 nodes per simulation**.

### 3.1 Deterministic Geometric Node Selection
A naive random sampling would completely miss critical physical phenomena. For example, 99.4% of the fluid domain is ambient, while extreme thermal gradients exist only in a microscopic boundary layer near the chip. To address this, a **Deterministic Geometric Node Selection** algorithm was implemented:

* **Tier 0 (Chip Wall, $y < 0.0001m$):** **100% force-included**. The chip is the heat source. Without forcing these nodes, the model would never learn the high-temperature gradient dynamics.
* **Tier 1 (Jet Core / Inlet, $r < D_m/2$ & $y > y_{max} - 5mm$):** **100% force-included**. The inlet carries the velocity boundary condition. 
* **Tier 2 (Bulk Fluid):** The remaining node budget (up to 250,000) is sampled uniformly using a fixed random seed (`seed=42`). 

**Crucial Design Decision:** The sampling is *deterministic* and based purely on spatial geometry, not on simulation-specific physical values (like velocity fields). This guarantees that every simulation with the same $H/D$ geometry outputs features at the **exact same physical coordinates**, ensuring alignment with the static graph structural templates generated later.

### 3.2 Feature Selection and Physical Priors
To make the learning process easier, raw inputs are transformed into 16 physics-informed features (3 Global, 7 Spatial, 1 Skewed, 5 Binary):
* **Gradient Proxies:** Variables like `temp_gradient_proxy` ($Heat Flux / Radius$) and `inv_radius` ($1 / Radius$) are manually engineered to help the neural network understand radial heat dissipation without having to infer it purely from $X$ and $Z$.
* **Stagnation Flag:** A smooth Gaussian stagnation encoding ($e^{-(r/D_m)^2}$) identifies the high-turbulence impingement zone directly below the jet.
* **Pressure Normalization:** Pressure exhibits extreme dynamic ranges with negative gauge values. It is stabilized using a signed log transformation: $sign(p) \cdot \ln(1 + |p|)$.
* **Boundary Condition Velocity (`bc_velocity`):** Normalized and spatially decayed ($vel\_norm \cdot e^{-r/D_m}$) to ensure a continuous representation of the inlet jet.

---

## 4. Graph Generation: Dimensionality Reduction (`dimen_red.py`)

Even at 250,000 nodes, running full $O(N^2)$ message passing across the entire domain is too expensive. The project utilizes a **Latent Graph** structure (Encoder-Processor-Decoder paradigm).

### 4.1 Physics-Aware FAISS KMeans Clustering
Instead of operating on 250k nodes directly, the geometry is clustered into **3,000 Latent Nodes**. 
* **Physics-Aware Weighting:** Standard K-Means would place latent nodes uniformly. We weight the K-Means sampling probability by $1/radius$, forcing FAISS to allocate more latent nodes at the jet center (stagnation zone) where turbulence and thermal gradients are highest, and fewer nodes in the ambient distant fluid.

### 4.2 Bipartite Edge Construction
Using `faiss.IndexFlatL2`, bipartite edges are constructed:
* **Fine → Latent ($k=8$):** 250k fine nodes map up to their 8 nearest latent nodes.
* **Latent ↔ Latent ($k=32$):** The 3,000 latent nodes interact heavily with each other.
* **Latent → Fine ($k=8$):** Latent nodes broadcast information back down to the fine mesh.

**Edge Attributes:** For every edge, the network calculates distance and relative Cartesian vectors: `[distance, dx, dy, dz]`. This ensures the message passing is strictly translation-invariant and geometrically grounded.

---

## 5. Model Selection: Jet Latent GNN (`LatentGNN.py`)

The neural architecture chosen is a **Heterogeneous Latent Graph Neural Network**, specifically engineered for fluid dynamics.

### 5.1 Architecture Flow (Encoder-Processor-Decoder)
1. **Encoder:** A 3-layer MLP encodes the 16 physical features of the 250k Fine nodes into a 128-dimensional hidden space.
2. **Up-Convolution:** A `GATv2Conv` (Graph Attention Network v2) aggregates Fine node features into the 3,000 Latent nodes. Attention is crucial here, as the network learns which fine nodes (e.g., wall vs. ambient) matter most to the latent representation.
3. **Latent Processor:** 4 layers of `GATv2Conv` operate *only* on the 3,000 latent nodes. This is where the long-range physics (fluid flow, pressure waves) are solved. Because $N=3000$, this step is computationally trivial. Residual connections and LayerNorms prevent gradient vanishing.
4. **Down-Convolution:** Processed latent states are broadcast back to the 250k Fine nodes via `GATv2Conv`.
5. **Decoder:** A 3-layer MLP maps the resulting 128-dim features to the 5 physical targets: $T, P, U_{vel}, V_{vel}, W_{vel}$.

### 5.2 Global Context Broadcasting
Navier-Stokes solutions are highly dependent on global boundary conditions. To prevent the model from having to "pass messages" across the entire graph just to realize the input power is 60W, a `param_mlp` creates an embedding of the global conditions. 
* This global context is **added directly to every latent node** during processing. 
* Furthermore, global mean/max pooling over the latent graph provides a macro-summary of the fluid state, which is fed back into the latent nodes at every layer.

---

## 6. Training Strategies & Memory Management (`train.py` & `gnn.md`)

Training a graph model of this magnitude requires advanced system-level engineering.

### 6.1 Zero-Copy Template Fusion
Loading a 2.26 GB structural graph for every simulation file would destroy data-loader performance. 
* **Decision:** The bipartite graphs (edges) for $H/D=4, 5, 6$ are pre-computed in `dimen_red.py` and saved as static templates. 
* During training, these structural templates are loaded into VRAM **once**. The DataLoader merely streams the `x` (features) and `y` (targets) arrays, injecting them dynamically into the template's memory space: `graph['fine'].x = x_in`. This makes data-loading instantaneous.

### 6.2 Gradient Accumulation
Even with the latent graph, storing the activation maps for 250k nodes maxes out VRAM. 
* **Decision:** The batch size is forced to `1`. To achieve statistical stability equivalent to `batch_size=8`, **Gradient Accumulation** is used. Loss is divided by 8, and `.step()` is only called every 8 iterations.

### 6.3 Physics-Informed Weighted Loss
A standard MSE/MAE loss would cause the model to perform beautifully in the ambient fluid (which is easy to predict) and poorly in the highly turbulent impingement zone.
* **Decision:** A custom `physics_weighted_loss` uses Smooth L1 (Huber) Loss to prevent massive pressure spikes from exploding the gradients. Crucially, the loss is **multiplied by 5.0x** for any node located in the stagnation zone (`Stagnation_Flag == 1`). This forces the optimizer to heavily penalize errors in the hardest-to-predict physical region.

---

## 7. Inference and Pipeline Execution

The final system operates in a strict sequence:
1. `FeatureEngineering.py`: Extracts raw CFD, deterministically samples 250k nodes, calculates physical priors, normalizes targets.
2. `dimen_red.py`: Runs physics-aware FAISS KMeans, creates Latent nodes, builds up/down/latent message-passing edges, exports Structural Templates.
3. `train.py`: Bootstraps Zero-Copy templates, streams features, trains `JetLatentGNN` with physics-weighted loss and gradient accumulation.
4. `evaluate_surrogate.py` (Inference): Loads arbitrary unseen boundary conditions, perfectly interpolates the physical fields, and executes Matplotlib 3D spatial plotting in fractions of a second.
