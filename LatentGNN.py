import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class MLP(nn.Module):
    """A standard Multi-Layer Perceptron for independent node processing."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_channels, hidden_channels))
        layers.append(nn.GELU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(nn.GELU())
            
        layers.append(nn.Linear(hidden_channels, out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class JetLatentGNN(nn.Module):
    def __init__(self, in_features=12, hidden_features=128, out_features=5, latent_layers=4):
        super().__init__()
        
        print("Initializing JetLatentGNN...")
        print(f" -> Input Features: {in_features}")
        print(f" -> Hidden Dimension: {hidden_features}")
        print(f" -> Output Targets: {out_features}")
        
        # 1. Independent Fine-Node Encoder
        self.encoder = MLP(in_features, hidden_features, hidden_features, num_layers=3)
        
        # 2. Bipartite Up-Pool (Fine -> Latent)
        # SAGEConv requires (source_channels, target_channels) for bipartite graphs
        self.up_conv = SAGEConv((hidden_features, hidden_features), hidden_features)
        
        # 3. Latent Space Processor (Latent -> Latent)
        self.processor_layers = nn.ModuleList([
            SAGEConv(hidden_features, hidden_features) for _ in range(latent_layers)
        ])
        
        # 4. Bipartite Down-Pool (Latent -> Fine)
        self.down_conv = SAGEConv((hidden_features, hidden_features), hidden_features)
        
        # 5. Independent Fine-Node Decoder
        # We multiply by 2 because we use a skip connection (original encoded + down-pooled)
        self.decoder = MLP(hidden_features * 2, hidden_features, out_features, num_layers=3)

    def forward(self, data):
        """
        data: The HeteroData object from dimen_red.py fused with dynamic DataLoader features.
        """
        # Extract features and edge matrices
        x_fine = data['fine'].x
        
        # Bug 3 Fix: Extract edge attributes (distances) and convert to inverse-weights
        # This gives spatially closer nodes higher influence in the message passing.
        edge_index_up = data['fine', 'maps_to', 'latent'].edge_index
        edge_attr_up  = data['fine', 'maps_to', 'latent'].edge_attr
        w_up = 1.0 / (edge_attr_up.squeeze() + 1e-6)
        
        edge_index_down = data['latent', 'maps_to', 'fine'].edge_index
        edge_attr_down  = data['latent', 'maps_to', 'fine'].edge_attr
        w_down = 1.0 / (edge_attr_down.squeeze() + 1e-6)
        
        edge_index_latent = data['latent', 'interacts_with', 'latent'].edge_index
        edge_attr_latent  = data['latent', 'interacts_with', 'latent'].edge_attr
        w_lat = 1.0 / (edge_attr_latent.squeeze() + 1e-6)
        
        # ==========================================
        # 1. ENCODE
        # ==========================================
        h_fine_encoded = self.encoder(x_fine)
        
        # Initialize latent node features as zeros (they are empty containers waiting for data)
        num_latent = data['latent'].pos.size(0)
        h_latent = torch.zeros((num_latent, h_fine_encoded.size(1)), device=h_fine_encoded.device)
        
        # ==========================================
        # 2. UP-POOL (Fine -> Latent)
        # ==========================================
        # Pass messages from fine (source) to latent (target) weighted by distance
        h_latent = self.up_conv((h_fine_encoded, h_latent), edge_index_up, edge_weight=w_up)
        h_latent = F.gelu(h_latent)
        
        # ==========================================
        # 3. PROCESS LATENT MACRO-PHYSICS
        # ==========================================
        # Pass messages across the latent graph to simulate pressure/momentum waves
        for conv in self.processor_layers:
            h_latent_new = conv(h_latent, edge_index_latent, edge_weight=w_lat)
            h_latent_new = F.gelu(h_latent_new)
            h_latent = h_latent + h_latent_new # Residual/Skip connection prevents vanishing gradients
            
        # ==========================================
        # 4. DOWN-POOL (Latent -> Fine)
        # ==========================================
        # Pass messages from latent (source) back to fine (target) weighted by distance
        h_fine_decoded = self.down_conv((h_latent, h_fine_encoded), edge_index_down, edge_weight=w_down)
        h_fine_decoded = F.gelu(h_fine_decoded)
        
        # ==========================================
        # 5. DECODE
        # ==========================================
        # Concatenate the deeply processed fluid physics with the raw encoded geometry 
        # so the network remembers exactly where the wall and stagnation zones are.
        out = self.decoder(torch.cat([h_fine_decoded, h_fine_encoded], dim=-1))
        
        return out

# Quick test if you run this script directly
if __name__ == "__main__":
    from torch_geometric.data import HeteroData
    
    model = JetLatentGNN(in_features=12) # Confirm: 4G + 6S + 1K + 1B = 12 inputs, 5 outputs (T, P, U, V, W)
    
    # Number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel instantiated successfully with {total_params:,} trainable parameters.")
    
    # Smoke test with dummy HeteroData
    print("\nRunning forward-pass smoke test...")
    num_fine, num_latent = 1000, 100  # small for testing
    
    data = HeteroData()
    data['fine'].x   = torch.randn(num_fine, 12)
    data['fine'].pos  = torch.randn(num_fine, 3)
    data['latent'].pos = torch.randn(num_latent, 3)
    
    # Dummy edges: fine->latent (up), latent->fine (down), latent->latent
    data['fine', 'maps_to', 'latent'].edge_index = torch.randint(0, num_fine, (2, num_fine * 3))
    data['fine', 'maps_to', 'latent'].edge_index[1] = torch.randint(0, num_latent, (num_fine * 3,))
    data['fine', 'maps_to', 'latent'].edge_attr  = torch.rand(num_fine * 3, 1) # Added Bug 3
    
    data['latent', 'maps_to', 'fine'].edge_index = torch.flip(
        data['fine', 'maps_to', 'latent'].edge_index, dims=[0]
    )
    data['latent', 'maps_to', 'fine'].edge_attr  = torch.rand(num_fine * 3, 1) # Added Bug 3
    
    src_lat = torch.randint(0, num_latent, (num_latent * 15,))
    tgt_lat = torch.randint(0, num_latent, (num_latent * 15,))
    data['latent', 'interacts_with', 'latent'].edge_index = torch.stack([src_lat, tgt_lat])
    data['latent', 'interacts_with', 'latent'].edge_attr  = torch.rand(num_latent * 15, 1) # Added Bug 3
    
    out = model(data)
    print(f"Output shape: {out.shape}  (expected: [{num_fine}, 5])")
    assert out.shape == (num_fine, 5), f"Shape mismatch! Got {out.shape}"
    print("✓ Smoke test passed!")