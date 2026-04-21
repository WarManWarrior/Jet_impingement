import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool


# ─────────────────────────────────────────────────
#  MLP BLOCK
# ─────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────
#  LATENT GNN MODEL
# ─────────────────────────────────────────────────
class JetLatentGNN(nn.Module):
    def __init__(self, in_features=18, hidden_features=128, out_features=5, latent_layers=4):
        super().__init__()

        print("Initializing JetLatentGNN v3 (Physics-aware, peak-preserving)")
        print(f" -> Input Features  : {in_features}")
        print(f" -> Hidden Dimension: {hidden_features}")

        self.dropout = 0.1

        # ── Encoder ───────────────────────────────
        self.encoder = MLP(in_features, hidden_features, hidden_features, num_layers=3)

        # ── Fine → Latent ─────────────────────────
        self.up_conv = GATv2Conv(
            (hidden_features, hidden_features),
            hidden_features,
            heads=1,
            edge_dim=4,   # updated
            add_self_loops=False,
            concat=False
        )

        # ── Latent Processor ──────────────────────
        self.processor_layers = nn.ModuleList([
            GATv2Conv(
                hidden_features,
                hidden_features,
                heads=1,
                edge_dim=4,   # updated
                add_self_loops=False,
                concat=False
            )
            for _ in range(latent_layers)
        ])

        # ── LayerNorms ────────────────────────────
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_features) for _ in range(latent_layers)
        ])

        # ── Global Context MLPs ───────────────────
        self.global_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_features * 2, hidden_features),  # mean+max
                nn.LayerNorm(hidden_features),
                nn.GELU(),
                nn.Linear(hidden_features, hidden_features)
            )
            for _ in range(latent_layers)
        ])

        # ── Parameter Node (global conditioning) ──
        self.param_mlp = MLP(6, 128, hidden_features, num_layers=3)  # deeper

        # ── Latent → Fine ─────────────────────────
        self.down_conv = GATv2Conv(
            (hidden_features, hidden_features),
            hidden_features,
            heads=1,
            edge_dim=4,   # updated
            add_self_loops=False,
            concat=False
        )

        # ── Decoder ───────────────────────────────
        self.decoder = MLP(
            hidden_features * 2 + hidden_features,
            hidden_features,
            out_features,
            num_layers=3
        )

    # ─────────────────────────────────────────────
    #  FORWARD
    # ─────────────────────────────────────────────
    def forward(self, data):
        x_fine = data['fine'].x

        edge_index_up   = data['fine', 'maps_to', 'latent'].edge_index
        edge_index_down = data['latent', 'maps_to', 'fine'].edge_index
        edge_index_lat  = data['latent', 'interacts_with', 'latent'].edge_index

        edge_attr_up   = data['fine', 'maps_to', 'latent'].edge_attr
        edge_attr_down = data['latent', 'maps_to', 'fine'].edge_attr
        edge_attr_lat  = data['latent', 'interacts_with', 'latent'].edge_attr

        # ── Encode fine nodes ─────────────────────
        h_fine = self.encoder(x_fine)

        num_latent = data['latent'].pos.size(0)
        h_latent = torch.zeros((num_latent, h_fine.size(1)), device=h_fine.device)

        # ── Fine → Latent ─────────────────────────
        h_latent = self.up_conv((h_fine, h_latent), edge_index_up, edge_attr_up)
        h_latent = F.gelu(h_latent)

        # ── Parameter embedding ───────────────────
        global_params = x_fine[0, :6]
        param_emb = self.param_mlp(global_params.unsqueeze(0))

        batch = torch.zeros(num_latent, dtype=torch.long, device=h_latent.device)

        # ── Latent Processing ─────────────────────
        for i, (conv, global_mlp) in enumerate(zip(self.processor_layers, self.global_mlps)):

            h_new = conv(h_latent, edge_index_lat, edge_attr_lat)
            h_new = F.gelu(h_new)

            # Residual
            h_new = h_latent + h_new

            # Normalize
            h_new = self.norms[i](h_new)

            # Dropout
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)

            # Global context (mean + max pooling)
            g_mean = global_mean_pool(h_new, batch)
            g_max  = global_max_pool(h_new, batch)
            g_summary = torch.cat([g_mean, g_max], dim=-1)

            g_context = global_mlp(g_summary)

            # Broadcast global + parameter context
            h_latent = h_new + g_context[batch] + param_emb[batch]

        # ── Latent → Fine ─────────────────────────
        h_fine_decoded = self.down_conv((h_latent, h_fine), edge_index_down, edge_attr_down)
        h_fine_decoded = F.gelu(h_fine_decoded)

        # ── Combine features ──────────────────────
        combined = torch.cat([
            h_fine_decoded,
            h_fine,
            param_emb.repeat(h_fine.size(0), 1)
        ], dim=-1)

        #Stabilize decoder input
        combined = F.layer_norm(combined, combined.shape[-1:])

        # ── Decode ────────────────────────────────
        out = self.decoder(combined)

        return out


# ─────────────────────────────────────────────────
#  SMOKE TEST
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    from torch_geometric.data import HeteroData
    import numpy as np

    print("\nRunning smoke test (v3)...")

    model = JetLatentGNN(in_features=18, hidden_features=64, out_features=5, latent_layers=2)

    num_fine, num_latent = 1000, 50

    data = HeteroData()
    data['fine'].x   = torch.randn(num_fine, 18)
    data['fine'].pos = torch.randn(num_fine, 3)
    data['latent'].pos = torch.randn(num_latent, 3)

    data['fine', 'maps_to', 'latent'].edge_index = torch.randint(0, num_latent, (2, num_fine * 8))
    data['fine', 'maps_to', 'latent'].edge_attr  = torch.randn(num_fine * 8, 4)

    data['latent', 'maps_to', 'fine'].edge_index = torch.randint(0, num_fine, (2, num_fine * 8))
    data['latent', 'maps_to', 'fine'].edge_attr  = torch.randn(num_fine * 8, 4)

    src_lat = torch.randint(0, num_latent, (num_latent * 31,))
    dst_lat = torch.randint(0, num_latent, (num_latent * 31,))
    data['latent', 'interacts_with', 'latent'].edge_index = torch.stack([src_lat, dst_lat])
    data['latent', 'interacts_with', 'latent'].edge_attr  = torch.randn(num_latent * 31, 4)

    out = model(data)

    print(f"Output shape: {out.shape}  (expected: [{num_fine}, 5])")
    assert out.shape == (num_fine, 5)

    print("Smoke test passed!")