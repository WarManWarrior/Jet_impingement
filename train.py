import os
import re
import glob
import random
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np

# Import your custom modules
from FeatureEngineering import JetImpingementDataset
from LatentGNN import JetLatentGNN

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = r"D:\data\JET"
EPOCHS = 100
ACCUMULATION_STEPS = 8   # Simulates batch_size=8 to prevent OOM
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using compute device: {DEVICE}")

# ==========================================
# 1. DISCOVER PER-SIMULATION FILES
# ==========================================
print("\nDiscovering per-simulation .pt files...")

train_sim_dir = os.path.join(DATA_DIR, 'train_sims')
val_sim_dir   = os.path.join(DATA_DIR, 'val_sims')

train_sim_files = sorted(glob.glob(os.path.join(train_sim_dir, '*.pt')))
val_sim_files   = sorted(glob.glob(os.path.join(val_sim_dir,   '*.pt')))

print(f"Found {len(train_sim_files)} train simulations, {len(val_sim_files)} val simulations.")

# ==========================================
# 2. PRE-LOAD GRAPH TEMPLATES (Zero-Copy)
# ==========================================
print("Pre-loading Structural Graph Templates (HD 4, 5, 6)...")
templates = {}
for hd in [4, 5, 6]:
    template_path = os.path.join(DATA_DIR, f'processed_graph_sim_HD{hd}_001.pt')
    if os.path.exists(template_path):
        graph = torch.load(template_path, weights_only=False)
        # Move structural edges to GPU once permanently. We will inject features later.
        graph = graph.to(DEVICE)
        templates[hd] = graph
        print(f"  ✓ HD={hd} template loaded ({graph['fine'].pos.shape[0]:,} fine nodes)")
    else:
        print(f"  ⚠ Warning: Template for HD={hd} not found at {template_path}.")

# ==========================================
# 3. INITIALIZE MODEL & OPTIMIZER
# ==========================================
# in_features is strictly 12 (4 Global + 6 Spatial + 1 Skewed + 1 Binary)
model = JetLatentGNN(in_features=12, hidden_features=128, out_features=5, latent_layers=4).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {total_params:,}")

# AdamW with weight decay helps prevent the model from overfitting the ambient fluid
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# Reduce LR if validation loss stalls
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# ==========================================
# 4. PHYSICS-INFORMED LOSS FUNCTION
# ==========================================
def physics_weighted_loss(predictions, targets, inputs, aux_targets=None, return_breakdown=False):
    """
    Computes Smooth L1 Loss but penalizes stagnation zone errors heavily.
    Plus an auxiliary term for dimensionless physics targets (Bug 4).
    """
    # Smooth L1 prevents massive pressure gradients from exploding the loss
    base_loss = F.smooth_l1_loss(predictions, targets, reduction='none')
    
    # Extract stagnation flag (0.0 or 1.0) at index 11
    stagnation_flag = inputs[:, 11].unsqueeze(1) # Shape: [N, 1]
    
    # Create weight matrix: 1.0 everywhere, but 5.0 where Stagnation_Flag == 1
    weights = torch.ones_like(base_loss)
    weights = weights + (stagnation_flag * 4.0) # 1.0 + 4.0 = 5.0 multiplier
    
    weighted_loss = base_loss * weights
    mean_total = weighted_loss.mean()
    
    # Bug 4 Fix: Add Auxiliary Physics Loss (encourage Theta_norm conservation)
    if aux_targets is not None:
        # Theta_norm is at sim_ds.A[:, 0]
        preds_theta = predictions[:, 0:1] # temperature target correlates with theta
        aux_loss = F.smooth_l1_loss(preds_theta, aux_targets[:, 0:1])
        mean_total = mean_total + 0.1 * aux_loss
    
    if return_breakdown:
        breakdown = {
            'L_T': weighted_loss[:, 0].mean().item(),
            'L_P': weighted_loss[:, 1].mean().item(),
            'L_U': weighted_loss[:, 2].mean().item(),
            'L_V': weighted_loss[:, 3].mean().item(),
            'L_W': weighted_loss[:, 4].mean().item()
        }
        return mean_total, breakdown
        
    return mean_total

# ==========================================
# 5. MAIN TRAINING LOOP
# ==========================================
print("\n" + "="*60)
print("🚀 INITIATING DIGITAL TWIN TRAINING SEQUENCE")
print("="*60)

best_val_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss = 0
    optimizer.zero_grad()
    
    # Shuffle training files each epoch for stochastic ordering
    random.shuffle(train_sim_files)
    
    loop = tqdm(train_sim_files, desc=f"Epoch {epoch:02d}/{EPOCHS} [Train]")
    for i, sim_path in enumerate(loop):
        
        # Load one simulation's pre-scaled dataset
        sim_ds = torch.load(sim_path, weights_only=False)
        x_in = sim_ds.X.squeeze(0).to(DEVICE)    # shape: [~100k, 12]
        t_in = sim_ds.T.squeeze(0).to(DEVICE)    # shape: [~100k, 5]
        a_in = sim_ds.A.squeeze(0).to(DEVICE)    # shape: [~100k, 2] Bug 4 Fix: load aux targets
        del sim_ds
        
        # Validate feature count
        assert x_in.shape[1] == 12, f"Feature count mismatch: got {x_in.shape[1]}, expected 12"
        
        # Extract H/D from the filename (e.g., 'H4/Vel_5_Pow_100.pt' → hd=4)
        hd_match = re.search(r'H(\d+)', os.path.basename(sim_path))
        if hd_match:
            hd_val = int(hd_match.group(1))
        else:
            # Fallback: try parsing from directory structure
            hd_match = re.search(r'[/\\]H(\d+)[/\\]', sim_path)
            hd_val = int(hd_match.group(1)) if hd_match else 4  # safe default
        
        graph = templates[hd_val]
        
        # Inject dynamic features into the static structural template
        graph['fine'].x = x_in
        graph['fine'].y = t_in
        
        # FORWARD PASS
        preds = model(graph)
        
        # COMPUTE PHYSICS LOSS (Bug 4 Fix: include aux_targets)
        loss, bkd = physics_weighted_loss(preds, t_in, x_in, aux_targets=a_in, return_breakdown=True)
        
        # GRADIENT ACCUMULATION
        loss = loss / ACCUMULATION_STEPS
        loss.backward()
        
        # Step the optimizer only after hitting the accumulation threshold
        if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_sim_files):
            # Clip gradients to max norm 1.0 to prevent explosion from extreme pressure spikes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
        total_train_loss += loss.item() * ACCUMULATION_STEPS
        
        # Create detailed tracker log string
        loop.set_postfix({
            'loss': f"{loss.item() * ACCUMULATION_STEPS:.4f}",
            'T': f"{bkd['L_T']:.3f}", 'P': f"{bkd['L_P']:.3f}", 
            'U': f"{bkd['L_U']:.3f}", 'V': f"{bkd['L_V']:.3f}", 'W': f"{bkd['L_W']:.3f}"
        })

    avg_train_loss = total_train_loss / len(train_sim_files)
    
    if str(DEVICE) == 'cuda':
        vram_alloc = torch.cuda.memory_allocated(DEVICE) / (1024**3)
        vram_peak = torch.cuda.max_memory_allocated(DEVICE) / (1024**3)
        print(f"   ► VRAM: Alloc={vram_alloc:.2f}GB / Peak={vram_peak:.2f}GB")
    
    # --- VALIDATION PHASE ---
    model.eval()
    total_val_loss = 0
    
    with torch.no_grad():
        val_loop = tqdm(val_sim_files, desc=f"Epoch {epoch:02d}/{EPOCHS} [Val  ]")
        for sim_path in val_loop:
            sim_ds = torch.load(sim_path, weights_only=False)
            x_in = sim_ds.X.squeeze(0).to(DEVICE)
            t_in = sim_ds.T.squeeze(0).to(DEVICE)
            a_in = sim_ds.A.squeeze(0).to(DEVICE) # Bug 4 Fix: load aux targets
            del sim_ds
            
            # Extract H/D from filename
            hd_match = re.search(r'H(\d+)', os.path.basename(sim_path))
            if hd_match:
                hd_val = int(hd_match.group(1))
            else:
                hd_match = re.search(r'[/\\]H(\d+)[/\\]', sim_path)
                hd_val = int(hd_match.group(1)) if hd_match else 4
            
            graph = templates[hd_val]
            graph['fine'].x = x_in
            
            preds = model(graph)
            # VALIDATION LOSS (Bug 4 Fix: include aux_targets)
            loss, bkd = physics_weighted_loss(preds, t_in, x_in, aux_targets=a_in, return_breakdown=True)
            
            total_val_loss += loss.item()
            val_loop.set_postfix({
                'v_loss': f"{loss.item():.4f}",
                'T': f"{bkd['L_T']:.3f}", 'P': f"{bkd['L_P']:.3f}", 
                'U': f"{bkd['L_U']:.3f}", 'V': f"{bkd['L_V']:.3f}", 'W': f"{bkd['L_W']:.3f}"
            })
            
    avg_val_loss = total_val_loss / len(val_sim_files)
    scheduler.step(avg_val_loss)
    
    print(f"📊 Epoch {epoch} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    # Save the Best Model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        save_path = os.path.join(DATA_DIR, "best_jet_surrogate_model.pth")
        torch.save(model.state_dict(), save_path)
        print(f"⭐ New Best Model Saved -> {save_path}")

print("\n🎉 Training Complete! The surrogate digital twin is ready for inference.")
