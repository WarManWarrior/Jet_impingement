import os
import glob
import torch

class JetImpingementDataset:
    pass

DATA_DIR = r"D:\data\JET"

def inspect_data():
    train_sim_dir = os.path.join(DATA_DIR, 'train_sims')
    val_sim_dir = os.path.join(DATA_DIR, 'val_sims')
    
    train_files = sorted(glob.glob(os.path.join(train_sim_dir, '*.pt')))
    val_files = sorted(glob.glob(os.path.join(val_sim_dir, '*.pt')))
    
    print(f"Found {len(train_files)} train files and {len(val_files)} val files.")
    
    if not train_files:
        print("No training files found.")
        return
        
    sample_file = train_files[0]
    print(f"\nInspecting sample file: {sample_file}")
    
    # Load dataset object. weights_only=False is needed if it's a structural class
    sim_ds = torch.load(sample_file, weights_only=False)
    
    print(f"\nDataset Attributes:")
    print(f"Type: {type(sim_ds)}")
    
    x_in = None
    if hasattr(sim_ds, 'X'):
        x_in = sim_ds.X.squeeze(0)
        print(f"X (Input Features) Shape: {x_in.shape}")
        print(f"X - Min: {x_in.min().item():.4f}, Max: {x_in.max().item():.4f}, Mean: {x_in.mean().item():.4f}, NaN count: {torch.isnan(x_in).sum().item()}")
    else:
        print("No 'X' attribute found.")
    
    t_in = None
    if hasattr(sim_ds, 'T'):
        t_in = sim_ds.T.squeeze(0)
        print(f"T (Target Features) Shape: {t_in.shape}")
        print(f"T - Min: {t_in.min().item():.4f}, Max: {t_in.max().item():.4f}, Mean: {t_in.mean().item():.4f}, NaN count: {torch.isnan(t_in).sum().item()}")
    else:
        print("No 'T' attribute found.")
        
    print("\n--- Feature Details ---")
    if x_in is not None and x_in.dim() == 2:
        num_features = x_in.shape[1]
        print(f"\n[ Input Features: {num_features} total ]")
        print(f"{'Feature Name':<20} | {'Min':>10} | {'Max':>10} | {'Mean':>10} | {'Std':>10}")
        print("-" * 65)
        for i in range(num_features):
            std_val = x_in[:, i].to(torch.float32).std().item() if x_in[:, i].numel() > 1 else 0.0
            print(f"{f'Input Feature {i:02d}':<20} | {x_in[:, i].min().item():10.4f} | {x_in[:, i].max().item():10.4f} | {x_in[:, i].mean().item():10.4f} | {std_val:10.4f}")
            
    if t_in is not None and t_in.dim() == 2:
        num_targets = t_in.shape[1]
        target_names = ["T", "P", "Vx", "Vy", "Vz"]
        print(f"\n[ Target Features: {num_targets} total ]")
        print(f"{'Feature Name':<20} | {'Min':>10} | {'Max':>10} | {'Mean':>10} | {'Std':>10}")
        print("-" * 65)
        for i in range(num_targets):
            name = target_names[i] if i < len(target_names) else f"Target_{i}"
            std_val = t_in[:, i].to(torch.float32).std().item() if t_in[:, i].numel() > 1 else 0.0
            name_str = f"{name} (Feature {i})"
            print(f"{name_str:<20} | {t_in[:, i].min().item():10.4f} | {t_in[:, i].max().item():10.4f} | {t_in[:, i].mean().item():10.4f} | {std_val:10.4f}")

    # Inspect graph templates
    print("\n--- Inspecting Graph Templates ---")
    for hd in [4, 5, 6]:
        path = os.path.join(DATA_DIR, f'processed_graph_sim_HD{hd}_001.pt')
        if os.path.exists(path):
            try:
                graph = torch.load(path, weights_only=False)
                print(f"\nTemplate HD{hd}:")
                if 'fine' in graph and hasattr(graph['fine'], 'pos'):
                    print(f"  Fine Pos Shape: {graph['fine'].pos.shape}")
                if 'latent' in graph and hasattr(graph['latent'], 'pos'):
                    print(f"  Latent Pos Shape: {graph['latent'].pos.shape}")
                if 'fine' in graph and hasattr(graph['fine'], 'edge_index'):
                    print(f"  Fine Edge Index Shape: {graph['fine'].edge_index.shape}")
            except Exception as e:
                print(f"\nTemplate HD{hd} found but could not load: {e}")
        else:
            print(f"Template HD{hd} not found at {path}.")
            
    # Load Training Stats mapping if exists
    stats_path = os.path.join(DATA_DIR, 'training_stats.pt')
    if os.path.exists(stats_path):
        print("\n--- Training Stats (`training_stats.pt`) ---")
        stats = torch.load(stats_path, weights_only=True)
        for k, v in stats.items():
            if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {v}")
    else:
        print(f"\nTraining stats not found at {stats_path}")

if __name__ == "__main__":
    import numpy as np # import inside since stats might be np
    inspect_data()
