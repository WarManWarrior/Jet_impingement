"""
=============================================================================
  LATENT GNN — HETEROGENEOUS JET IMPINGEMENT SURROGATE

  CHANGES FROM PREVIOUS VERSION (ported from working v4 code):
    NEW  Virtual global node in each processor layer:
         Ported from v4 MGNBlock. After local GATv2Conv message passing, computes
         global_mean_pool over all latent nodes, transforms through a small MLP,
         and broadcasts back to every latent node. This is essential for pressure:
         pressure satisfies an elliptic PDE (Poisson equation) meaning it is
         globally coupled — a change at the inlet affects pressure everywhere
         simultaneously. Local message passing alone, regardless of depth, cannot
         propagate this. The global node provides an O(1) long-range path.

    FIX  in_features updated to 15 (was 14) for new bc_velocity feature.
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_mean_pool


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super().__init__()
        layers = [nn.Linear(in_channels, hidden_channels), nn.GELU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_channels, hidden_channels), nn.GELU()]
        layers.append(nn.Linear(hidden_channels, out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class JetLatentGNN(nn.Module):
    def __init__(self, in_features=16, hidden_features=128, out_features=5, latent_layers=4):
        super().__init__()
        print("Initializing JetLatentGNN v2 (Parameter Node + strong global)")
        print(f" -> Input Features  : {in_features}  (includes signed_dist_wall)")
        print(f" -> Hidden Dimension: {hidden_features}")

        self.encoder = MLP(in_features, hidden_features, hidden_features, num_layers=3)

        self.up_conv = GATv2Conv((hidden_features, hidden_features), hidden_features, heads=1, edge_dim=1, add_self_loops=False, concat=False)

        self.processor_layers = nn.ModuleList([
            GATv2Conv(hidden_features, hidden_features, heads=1, edge_dim=1, add_self_loops=False, concat=False)
            for _ in range(latent_layers)
        ])

        self.global_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_features, hidden_features),
                nn.LayerNorm(hidden_features),
                nn.GELU(),
                nn.Linear(hidden_features, hidden_features)
            ) for _ in range(latent_layers)
        ])

        # Parameter Node (continuous conditioning)
        self.param_mlp = MLP(6, 128, hidden_features, num_layers=2)

        self.down_conv = GATv2Conv((hidden_features, hidden_features), hidden_features, heads=1, edge_dim=1, add_self_loops=False, concat=False)

        self.decoder = MLP(hidden_features * 2 + hidden_features, hidden_features, out_features, num_layers=3)

    def forward(self, data):
        x_fine = data['fine'].x
        edge_index_up = data['fine', 'maps_to', 'latent'].edge_index
        edge_index_down = data['latent', 'maps_to', 'fine'].edge_index
        edge_index_latent = data['latent', 'interacts_with', 'latent'].edge_index

        edge_attr_up = data['fine', 'maps_to', 'latent'].edge_attr
        edge_attr_down = data['latent', 'maps_to', 'fine'].edge_attr
        edge_attr_latent = data['latent', 'interacts_with', 'latent'].edge_attr

        h_fine = self.encoder(x_fine)
        num_latent = data['latent'].pos.size(0)
        h_latent = torch.zeros((num_latent, h_fine.size(1)), device=h_fine.device)

        h_latent = self.up_conv((h_fine, h_latent), edge_index_up, edge_attr_up)
        h_latent = F.gelu(h_latent)

        # Parameter embedding (first 6 columns = global params)
        global_params = x_fine[0, :6]
        param_emb = self.param_mlp(global_params.unsqueeze(0))

        batch = torch.zeros(num_latent, dtype=torch.long, device=h_latent.device)

        for conv, global_mlp in zip(self.processor_layers, self.global_mlps):
            h_new = conv(h_latent, edge_index_latent, edge_attr_latent)
            h_new = F.gelu(h_new)
            h_new = h_latent + h_new

            g_summary = global_mean_pool(h_new, batch)
            g_context = global_mlp(g_summary)
            h_latent = h_new + g_context[batch] + param_emb[batch]

        h_fine_decoded = self.down_conv((h_latent, h_fine), edge_index_down, edge_attr_down)
        h_fine_decoded = F.gelu(h_fine_decoded)

        # Concatenate fine_decoded, fine_encoded, and repeated parameter embedding
        out = self.decoder(torch.cat([h_fine_decoded, h_fine, param_emb.repeat(h_fine.size(0), 1)], dim=-1))
        return out


# ─────────────────────────────────────────────────
#  SMOKE TEST
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    from torch_geometric.data import HeteroData
    import numpy as np

    print("\nRunning smoke test (v2 structure)...")
    model = JetLatentGNN(in_features=16, hidden_features=64, out_features=5, latent_layers=2)
    
    num_fine, num_latent = 1000, 50
    data = HeteroData()
    data['fine'].x   = torch.randn(num_fine, 16)
    data['fine'].pos = torch.randn(num_fine, 3)
    data['latent'].pos = torch.randn(num_latent, 3)
    
    data['fine', 'maps_to', 'latent'].edge_index = torch.randint(0, num_latent, (2, num_fine))
    data['fine', 'maps_to', 'latent'].edge_attr = torch.randn(num_fine, 1)
    
    data['latent', 'maps_to', 'fine'].edge_index = torch.randint(0, num_fine, (2, num_fine))
    data['latent', 'maps_to', 'fine'].edge_attr = torch.randn(num_fine, 1)
    
    src_lat = torch.randint(0, num_latent, (num_latent * 15,))
    dst_lat = torch.randint(0, num_latent, (num_latent * 15,))
    data['latent', 'interacts_with', 'latent'].edge_index = torch.stack([src_lat, dst_lat])
    data['latent', 'interacts_with', 'latent'].edge_attr = torch.randn(num_latent * 15, 1)

    out = model(data)
    print(f"Output shape: {out.shape}  (expected: [{num_fine}, 5])")
    assert out.shape == (num_fine, 5)
    print("✓ Smoke test passed!")