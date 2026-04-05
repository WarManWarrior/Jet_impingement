import os
import glob
import re
import h5py
import numpy as np
import torch
import faiss
from torch_geometric.data import HeteroData
from tqdm.auto import tqdm
import gc

def extract_reference_coordinates(folder_path="D:/data/JET/**/*.h5", target_hd=4.0):
    """
    Extracts raw X, Y, Z coordinates from ONE optimal mesh for a specific H/D ratio.
    """
    h5_files = glob.glob(folder_path, recursive=True)
    if not h5_files:
        raise FileNotFoundError("No .h5 files found to extract coordinates.")
    
    # Filter files for the specific target H/D
    target_files = []
    for fp in h5_files:
        hd_match = re.search(r'[/\\]H(\d+)[/\\]', fp)
        if hd_match and float(hd_match.group(1)) == target_hd:
            target_files.append(fp)
            
    if not target_files:
        raise ValueError(f"No files found for H/D = {target_hd}")
    
    reference_file = target_files[0]
    print(f"\n[HD={target_hd}] Extracting reference coordinates from: {reference_file}")
    
    with h5py.File(reference_file, 'r') as f:
        group_name = list(f.keys())[0]
        coords    = f[group_name]['Coordinates'][:]
        temp      = f[group_name]['Temperature'][:].ravel()
        vel_field = f[group_name]['Velocity'][:]
        
    # APPLY THE EXACT SAME SAMPLING AS FEATURE ENGINEERING (Deterministic Uniform)
    # This ensures every simulation with this geometry uses the same grid points.
    rng = np.random.default_rng(42) # Fixed seed across all geometries
    sel = rng.choice(np.arange(len(coords)), size=100_000, replace=False)
    sampled_coords = coords[sel]
    return np.float32(sampled_coords)

def generate_latent_mesh(coords: np.ndarray, num_latent_nodes: int = 10000):
    """
    Runs FAISS K-Means on GPU to extract the Latent node coordinates.
    """
    print(f"Running FAISS GPU K-Means to compress {len(coords):,} down to {num_latent_nodes:,} nodes...")
    try:
        # Install: pip install faiss-cpu
        # If you have a larger GPU (24GB+): conda install -c pytorch faiss-gpu
        kmeans = faiss.Kmeans(d=3, k=num_latent_nodes, niter=20, verbose=True, gpu=False)
    except Exception as e:
        print("GPU FAISS failed or unavailable, falling back to CPU FAISS...")
        kmeans = faiss.Kmeans(d=3, k=num_latent_nodes, niter=20, verbose=True)
        
    kmeans.train(coords)
    latent_coords = kmeans.centroids
    print(f"Latent nodes generated. Shape: {latent_coords.shape}")
    return latent_coords

def build_bipartite_edges_faiss(fine_coords: np.ndarray, latent_coords: np.ndarray):
    """
    Builds the PyG edge_index and edge_attr using FAISS IndexFlatL2.
    """
    print("Building up/down edge connections (k=3) using FAISS index...")
    # 1. Build FAISS index on Latent Coords
    index_latent = faiss.IndexFlatL2(3)
    index_latent.add(latent_coords)
    
    # 2. Query all fine nodes at once for upstream edges
    sq_distances, latent_indices = index_latent.search(fine_coords, k=3)
    edge_distances = np.sqrt(sq_distances.flatten())
    
    fine_indices = np.repeat(np.arange(len(fine_coords)), 3)
    edge_index_up = torch.tensor(np.vstack((fine_indices, latent_indices.flatten())), dtype=torch.long)
    edge_attr_up = torch.tensor(edge_distances, dtype=torch.float).unsqueeze(1)
    
    # 3. Down-edges are symmetric reverse
    edge_index_down = torch.flip(edge_index_up, dims=[0])
    edge_attr_down = edge_attr_up.clone()
    
    # 4. Latent Graph Edges (k=15)
    print("Building latent-to-latent edge connections (k=15)...")
    sq_dist_latent, latent_neighbors = index_latent.search(latent_coords, k=16)
    latent_neighbors = latent_neighbors[:, 1:]    # shape: [1000, 15]
    sq_dist_latent   = sq_dist_latent[:, 1:]      # shape: [1000, 15]
    latent_edge_dists = np.sqrt(sq_dist_latent.flatten())
    
    source_latent = np.repeat(np.arange(len(latent_coords)), 15)
    edge_index_latent = torch.tensor(np.vstack((source_latent, latent_neighbors.flatten())), dtype=torch.long)
    edge_attr_latent = torch.tensor(latent_edge_dists, dtype=torch.float).unsqueeze(1)
    
    return edge_index_up, edge_attr_up, edge_index_down, edge_attr_down, edge_index_latent, edge_attr_latent

def build_heterodata(X_fine: np.ndarray, Y_fine: np.ndarray, A_fine: np.ndarray, fine_coords: np.ndarray, latent_coords: np.ndarray,
                     e_up: torch.Tensor, a_up: torch.Tensor,
                     e_down: torch.Tensor, a_down: torch.Tensor,
                     e_lat: torch.Tensor, a_lat: torch.Tensor):
    """
    Assembles the complete PyG Heterogeneous Data Object.
    """
    data = HeteroData()
    
    # Node features
    data['fine'].x = torch.tensor(X_fine, dtype=torch.float)       # scaled
    data['fine'].y = torch.tensor(Y_fine, dtype=torch.float)       # scaled
    data['fine'].y_aux = torch.tensor(A_fine, dtype=torch.float)   # Shape: [~11.9M, 2]
    
    # Positions (Raw)
    data['fine'].pos = torch.tensor(fine_coords, dtype=torch.float)
    data['latent'].pos = torch.tensor(latent_coords, dtype=torch.float)
    
    # Edges & Attributes
    data['fine', 'maps_to', 'latent'].edge_index = e_up
    data['fine', 'maps_to', 'latent'].edge_attr  = a_up
    
    data['latent', 'maps_to', 'fine'].edge_index = e_down
    data['latent', 'maps_to', 'fine'].edge_attr  = a_down
    
    data['latent', 'interacts_with', 'latent'].edge_index = e_lat
    data['latent', 'interacts_with', 'latent'].edge_attr  = a_lat
    
    # Validation checks
    num_fine = len(fine_coords)
    num_latent = len(latent_coords)
    assert e_up.shape     == (2, num_fine * 3),           f"Up-edge count wrong: got {e_up.shape}"
    assert e_down.shape   == (2, num_fine * 3),           f"Down-edge count wrong: got {e_down.shape}"
    assert e_lat.shape    == (2, num_latent * 15),        f"Latent-edge count wrong: got {e_lat.shape}"
    
    total_bytes = (
        data['fine'].x.nbytes +
        data['fine'].y.nbytes +
        data['fine'].y_aux.nbytes +
        data['fine'].pos.nbytes +
        e_up.nbytes  + a_up.nbytes +
        e_down.nbytes + a_down.nbytes +
        e_lat.nbytes + a_lat.nbytes
    )
    print(f"Graph memory: {total_bytes / 1e9:.2f} GB")
    return data

if __name__ == "__main__":
    print("="*60)
    print("  HETEROGENEOUS GRAPH PRE-COMPUTATION PIPELINE (dimen_red)")
    print("="*60)
    
    hd_ratios = [4.0, 5.0, 6.0]
    num_latent_nodes = 10000  # CRITICAL: Up from 1000
    
    for target_hd in tqdm(hd_ratios, desc="Processing Geometries"):
        try:
            # 1. Coordinate Extraction
            try:
                fine_node_coords = extract_reference_coordinates(target_hd=target_hd)
            except Exception as e:
                print(f"\nError: {e}\nFalling back to dummy data for HD={target_hd}.")
                # Target: ~11.9 million nodes per geometry as specified in markdown
                fine_node_coords = np.random.rand(11900000, 3).astype(np.float32)
                
            num_fine = len(fine_node_coords)
            print(f"Total fine nodes for HD={target_hd}: {num_fine:,}")
            
            # 2. Latent Mapping
            latent_coords = generate_latent_mesh(fine_node_coords, num_latent_nodes=num_latent_nodes)
            
            # 3. Connectivity & Attributes
            e_up, a_up, e_down, a_down, e_lat, a_lat = build_bipartite_edges_faiss(fine_node_coords, latent_coords)
            
            # 4. Integrate Data
            print(f"\nCreating memory-mapped dummy X_fine, Y_fine, and A_fine representations...")
            # X_fine: 12 features | Y_fine: 5 target features | A_fine: 2 aux features
            X_fine_dummy = np.ones((num_fine, 12), dtype=np.float32) # Down from 13
            Y_fine_dummy = np.ones((num_fine, 5), dtype=np.float32)
            A_fine_dummy = np.ones((num_fine, 2), dtype=np.float32)
            
            gc.collect()
            
            graph_obj = build_heterodata(X_fine_dummy, Y_fine_dummy, A_fine_dummy, fine_node_coords, latent_coords,
                                         e_up, a_up, e_down, a_down, e_lat, a_lat)
            
            # 5. Export
            save_dir = r"D:\data\JET"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'processed_graph_sim_HD{int(target_hd)}_001.pt')
            torch.save(graph_obj, save_path)
            
            file_gb = os.path.getsize(save_path) / (1024 ** 3)
            print(f"\n✓ Graph Object Saved: {save_path}  ({file_gb:.2f} GB)\n")
            
            # Free massively heavy PyG graph object before next iteration
            del graph_obj, X_fine_dummy, Y_fine_dummy, fine_node_coords, latent_coords
            del e_up, a_up, e_down, a_down, e_lat, a_lat
            gc.collect()
            
        except Exception as e:
            print(f"Failed processing HD={target_hd}: {e}")
            
    print("Done! Review the exported .pt graph objects.")
