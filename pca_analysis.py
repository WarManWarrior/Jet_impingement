import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Global Config
DATA_DIR = r"D:\data\JET\FNO_Prepared"
SAVE_DIR = "pca_results"
os.makedirs(SAVE_DIR, exist_ok=True)

def run_pca_analysis():
    print("🚀 Initializing PCA Analysis Pipeline...", flush=True)
    

    # 1. Load Preprocessed Dataset
    files = glob.glob(os.path.join(DATA_DIR, "*.pt"))
    
    # ADD THIS LINE TO FILTER OUT THE SCALER:
    files = [f for f in files if "scaler.pt" not in os.path.basename(f)]
    
    print(f"Located {len(files)} processed matrices.", flush=True)
    
    if not files:
        print("❌ Error: No .pt files found in D:\\data\\JET\\FNO_Prepared", flush=True)
        return

    X = []
    vel_list, pow_list, D_list = [], [], []

    print("📊 Loading data and normalizing channels...", flush=True)
    for i, f in enumerate(files):
        if "scaler.pt" in f:
            continue
            
        data = torch.load(f)
        y = data["output"]  # [3, 64, 64, 64]

        # Normalize per channel (MANDATORY for PCA to prevent one field dominating)
        T = (y[0] - y[0].mean()) / (y[0].std() + 1e-6)
        P = (y[1] - y[1].mean()) / (y[1].std() + 1e-6)
        V = (y[2] - y[2].mean()) / (y[2].std() + 1e-6)

        y_norm = torch.stack([T, P, V])
        X.append(y_norm.flatten())

        vel_list.append(data["velocity"])
        pow_list.append(data["power"])
        D_list.append(data["D"])
        
        if (i+1) % 50 == 0:
            print(f"   Loaded {i+1}/{len(files)}...", flush=True)

    X = torch.stack(X).numpy()
    print(f"Data Matrix Shape: {X.shape} (Samples x Features)", flush=True)

    # 2. Perform Combined PCA
    print("\n🧬 Fitting PCA (10 components)...", flush=True)
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X)

    # 3. Plot Explained Variance
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, 11), pca.explained_variance_ratio_, marker='o', linestyle='--', color='b')
    plt.title("PCA Explained Variance (Multi-Physics Grid)")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Ratio")
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(SAVE_DIR, "explained_variance.png")
    plt.savefig(save_path)
    print(f"✅ Saved: {save_path}", flush=True)
    plt.close()

    # 4. Visualize PC1 Spatial Modes
    comp0 = pca.components_[0].reshape(3, 64, 64, 64)
    fields = ["Temperature", "Pressure", "Velocity"]
    
    plt.figure(figsize=(18, 5))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        # Slice at Z=32
        plt.imshow(comp0[i, :, :, 32], cmap='RdYlBu_r')
        plt.title(f"PC1 Mode: {fields[i]} (Z=32 slice)")
        plt.colorbar()
    
    save_path = os.path.join(SAVE_DIR, "pc1_modes_spatial.png")
    plt.savefig(save_path)
    print(f"✅ Saved: {save_path}", flush=True)
    plt.close()

    # 5. Scatter Plot PC1 vs PC2
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=vel_list, cmap='viridis', s=50, alpha=0.8)
    plt.colorbar(scatter, label='Inlet Velocity (m/s)')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("CFD Simulations in PCA Latent Space (Colored by Velocity)")
    plt.grid(True, alpha=0.2)
    save_path = os.path.join(SAVE_DIR, "pca_latent_space.png")
    plt.savefig(save_path)
    print(f"✅ Saved: {save_path}", flush=True)
    plt.close()

    # 6. Physical Parameter Correlation (Extended with Geometry)
    plt.figure(figsize=(20, 6))
    
    # 6A. PC1 vs Velocity
    plt.subplot(1, 3, 1)
    plt.scatter(vel_list, X_pca[:, 0], color='darkred', alpha=0.6)
    plt.xlabel("Inlet Velocity (m/s)")
    plt.ylabel("PC1 Score")
    plt.title("Correlation: Velocity vs PC1")
    plt.grid(True, alpha=0.2)

    # 6B. PC2 vs Power
    plt.subplot(1, 3, 2)
    plt.scatter(pow_list, X_pca[:, 1], color='darkblue', alpha=0.6)
    plt.xlabel("Input Power (W)")
    plt.ylabel("PC2 Score")
    plt.title("Correlation: Power vs PC2")
    plt.grid(True, alpha=0.2)
    
    # 6C. PC1 vs Diameter (NEW: Geometry Influence Check)
    plt.subplot(1, 3, 3)
    plt.scatter(D_list, X_pca[:, 0], color='darkgreen', alpha=0.6)
    plt.xlabel("Nozzle Diameter (D)")
    plt.ylabel("PC1 Score")
    plt.title("Correlation: Diameter vs PC1")
    plt.grid(True, alpha=0.2)
    
    save_path = os.path.join(SAVE_DIR, "param_correlations.png")
    plt.savefig(save_path)
    print(f"✅ Saved: {save_path}", flush=True)
    plt.close()

    # 7. Independent Pressure PCA (Secondary analysis)
    print("\n🔬 Fitting Independent Pressure PCA...", flush=True)
    X_p = []
    for f in files:
        data = torch.load(f)
        P = data["output"][1]
        P = (P - P.mean()) / (P.std() + 1e-6)
        X_p.append(P.flatten())
    
    X_p = torch.stack(X_p).numpy()
    pca_p = PCA(n_components=5)
    pca_p.fit(X_p)

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, 6), pca_p.explained_variance_ratio_, marker='s', color='green')
    plt.title("Independent Pressure PCA Variance")
    plt.xlabel("Component")
    plt.ylabel("Variance Ratio")
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(SAVE_DIR, "pressure_only_variance.png")
    plt.savefig(save_path)
    print(f"✅ Saved: {save_path}", flush=True)
    plt.close()

    print("\n✨ PCA Analysis Complete. All artifacts exported to /pca_results", flush=True)

if __name__ == "__main__":
    run_pca_analysis()
