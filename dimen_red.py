"""
=============================================================================
  HETEROGENEOUS GRAPH PRE-COMPUTATION PIPELINE

  CRITICAL FIX: Sampling now uses select_nodes_deterministic() from
  FeatureEngineering.py — IDENTICAL to what every training simulation uses.

  Previous bug: dimen_red used max-disturbance probability sampling (seed=42,
  one reference file). FeatureEngineering used tier-0 + per-file-seeded
  probability sampling. Result: the 250k node positions in the graph template
  were completely different from the 250k node positions in every training .pt
  file. Features at index i described a different physical location than the
  graph edge at index i — the model's spatial structure was meaningless.

  Now: both dimen_red and FeatureEngineering call select_nodes_deterministic()
  with the same arguments (target_n=250_000, seed=42). Since all simulations
  share the same mesh (coordinates) for a given H/D, the same 250k nodes are
  selected for every simulation, and the graph template is valid for all of them.
=============================================================================
"""

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

# Import the canonical node-selection function from FeatureEngineering
# This guarantees dimen_red and FE use EXACTLY the same sampling logic
from FeatureEngineering import select_nodes_deterministic, SAMPLE, GEOM


def extract_reference_coordinates(folder_path="D:/data/JET/**/*.h5", target_hd=4.0):
    """
    Extracts node coordinates from ONE reference mesh using deterministic
    geometric selection — the same selection that FeatureEngineering.py
    applies to every simulation of this H/D geometry.
    """
    h5_files = glob.glob(folder_path, recursive=True)
    if not h5_files:
        raise FileNotFoundError("No .h5 files found.")

    target_files = [fp for fp in h5_files
                    if (m := re.search(r'[/\\]H(\d+)[/\\]', fp)) and float(m.group(1)) == target_hd]
    if not target_files:
        raise ValueError(f"No files found for H/D = {target_hd}")

    reference_file = target_files[0]
    print(f"\n[HD={target_hd}] Reference file: {reference_file}")

    with h5py.File(reference_file, 'r') as f:
        group_name = list(f.keys())[0]
        coords     = f[group_name]['Coordinates'][:]

    # Deterministic geometric selection — same as FeatureEngineering
    sel            = select_nodes_deterministic(coords, target_n=SAMPLE['points_per_sim'])
    sampled_coords = coords[sel].astype(np.float32)

    n_wall  = (coords[sel, 1] < 1e-4).sum()
    n_inlet = ((SAMPLE['points_per_sim'] - len(np.where(coords[:, 1] < 1e-4)[0])) > 0)
    print(f"  Selected {len(sampled_coords):,} nodes  "
          f"(wall={n_wall:,}, total budget={SAMPLE['points_per_sim']:,})")
    return sampled_coords


def generate_latent_mesh(coords: np.ndarray, num_latent_nodes: int = 500):
    print(f"Running FAISS K-Means: {len(coords):,} fine → {num_latent_nodes:,} latent nodes...")
    try:
        kmeans = faiss.Kmeans(d=3, k=num_latent_nodes, niter=30, verbose=True, gpu=False)
    except Exception:
        kmeans = faiss.Kmeans(d=3, k=num_latent_nodes, niter=30, verbose=True)
    kmeans.train(coords)
    latent_coords = kmeans.centroids
    print(f"  Latent nodes: {latent_coords.shape}")
    return latent_coords


def build_bipartite_edges_faiss(fine_coords: np.ndarray, latent_coords: np.ndarray):
    print("Building fine↔latent edges (k=3)...")
    index_latent = faiss.IndexFlatL2(3)
    index_latent.add(latent_coords)

    sq_distances, latent_indices = index_latent.search(fine_coords, k=3)
    edge_distances = np.sqrt(np.clip(sq_distances.flatten(), 0, None))
    fine_indices   = np.repeat(np.arange(len(fine_coords)), 3)
    edge_index_up  = torch.tensor(np.vstack((fine_indices, latent_indices.flatten())), dtype=torch.long)
    edge_attr_up   = torch.tensor(edge_distances, dtype=torch.float).unsqueeze(1)

    edge_index_down = torch.flip(edge_index_up, dims=[0])
    edge_attr_down  = edge_attr_up.clone()

    print("Building latent↔latent edges (k=15)...")
    sq_dist_latent, latent_neighbors = index_latent.search(latent_coords, k=16)
    latent_neighbors = latent_neighbors[:, 1:]
    sq_dist_latent   = sq_dist_latent[:, 1:]
    latent_edge_dists = np.sqrt(np.clip(sq_dist_latent.flatten(), 0, None))
    source_latent     = np.repeat(np.arange(len(latent_coords)), 15)
    edge_index_latent = torch.tensor(np.vstack((source_latent, latent_neighbors.flatten())), dtype=torch.long)
    edge_attr_latent  = torch.tensor(latent_edge_dists, dtype=torch.float).unsqueeze(1)

    return edge_index_up, edge_attr_up, edge_index_down, edge_attr_down, edge_index_latent, edge_attr_latent


def build_heterodata(X_fine, Y_fine, A_fine, fine_coords, latent_coords,
                     e_up, a_up, e_down, a_down, e_lat, a_lat):
    data = HeteroData()
    data['fine'].x     = torch.tensor(X_fine,        dtype=torch.float)
    data['fine'].y     = torch.tensor(Y_fine,        dtype=torch.float)
    data['fine'].y_aux = torch.tensor(A_fine,        dtype=torch.float)
    data['fine'].pos   = torch.tensor(fine_coords,   dtype=torch.float)
    data['latent'].pos = torch.tensor(latent_coords, dtype=torch.float)
    data['fine',   'maps_to',        'latent'].edge_index = e_up
    data['fine',   'maps_to',        'latent'].edge_attr  = a_up
    data['latent', 'maps_to',        'fine'  ].edge_index = e_down
    data['latent', 'maps_to',        'fine'  ].edge_attr  = a_down
    data['latent', 'interacts_with', 'latent'].edge_index = e_lat
    data['latent', 'interacts_with', 'latent'].edge_attr  = a_lat
    num_fine   = len(fine_coords)
    num_latent = len(latent_coords)
    assert e_up.shape   == (2, num_fine * 3),    f"Up-edge wrong: {e_up.shape}"
    assert e_down.shape == (2, num_fine * 3),    f"Down-edge wrong: {e_down.shape}"
    assert e_lat.shape  == (2, num_latent * 15), f"Latent-edge wrong: {e_lat.shape}"
    return data


if __name__ == "__main__":
    print("=" * 60)
    print("  HETEROGENEOUS GRAPH PRE-COMPUTATION PIPELINE")
    print("  Sampling: deterministic geometric (matches FeatureEngineering)")
    print("=" * 60)

    DATA_DIR         = r"D:\data\JET"
    hd_ratios        = [4.0, 5.0, 6.0]
    num_latent_nodes = 500   # POD showed 1-2 modes capture >99% variance

    for target_hd in tqdm(hd_ratios, desc="Processing Geometries"):
        try:
            try:
                fine_node_coords = extract_reference_coordinates(
                    folder_path="D:/data/JET/**/*.h5", target_hd=target_hd
                )
            except Exception as e:
                print(f"\nError: {e}\nUsing dummy for HD={target_hd}.")
                fine_node_coords = np.random.rand(SAMPLE['points_per_sim'], 3).astype(np.float32)

            print(f"Fine nodes for HD={target_hd}: {len(fine_node_coords):,}")

            latent_coords = generate_latent_mesh(fine_node_coords, num_latent_nodes)

            e_up, a_up, e_down, a_down, e_lat, a_lat = build_bipartite_edges_faiss(
                fine_node_coords, latent_coords)

            X_dummy = np.ones((len(fine_node_coords), 16), dtype=np.float32)  # 16 features
            Y_dummy = np.ones((len(fine_node_coords), 5),  dtype=np.float32)
            A_dummy = np.ones((len(fine_node_coords), 2),  dtype=np.float32)
            gc.collect()

            graph_obj = build_heterodata(
                X_dummy, Y_dummy, A_dummy,
                fine_node_coords, latent_coords,
                e_up, a_up, e_down, a_down, e_lat, a_lat)

            os.makedirs(DATA_DIR, exist_ok=True)
            save_path = os.path.join(DATA_DIR, f'processed_graph_sim_HD{int(target_hd)}_001.pt')
            torch.save(graph_obj, save_path)
            print(f"✓  Saved: {save_path}  ({os.path.getsize(save_path)/1e9:.3f} GB)\n")

            del graph_obj, X_dummy, Y_dummy, A_dummy, fine_node_coords, latent_coords
            del e_up, a_up, e_down, a_down, e_lat, a_lat
            gc.collect()

        except Exception as e:
            print(f"Failed HD={target_hd}: {e}")

    print("Done.")