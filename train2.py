"""
=============================================================================
  TRAINING LOOP — JET IMPINGEMENT LATENT GNN SURROGATE
  PURE DATA-DRIVEN VERSION (NO BC ENFORCEMENT)

  Fixes applied vs original:
    1. signed_dist_x / signed_dist_z added to feature vector (18 → 20 features)
       — fixes near-zero U and W velocity predictions
    2. Stagnation weight boosted 3.0 → 15.0
       — fixes peak velocity under-prediction (model was capping at ~4 m/s)
    3. Velocity loss weights increased 1.5 → 3.0
       — forces model to prioritise getting U/V/W right
    4. EPOCHS 150 → 250, eta_min 1e-6 → 5e-5
       — val loss curves were still descending at epoch 149
    5. Pressure feature: assert updated for 20 features
=============================================================================
"""

import os
import re
import glob
import random
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np
import wandb

from FeatureEngineering import JetImpingementDataset, GEOM
from LatentGNN import JetLatentGNN

# ─────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────
DATA_DIR           = r"D:\data\JET"
EPOCHS             = 250          # Fix 4: was 150 — curves still descending
ACCUMULATION_STEPS = 8
LEARNING_RATE      = 3e-4
WEIGHT_DECAY       = 1e-4
GRAD_CLIP          = 1.0
IN_FEATURES        = 20          # Fix 1: was 18 — added signed_dist_x, signed_dist_z
DEVICE             = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

# ── wandb Initialization ─────────────────────────
wandb.init(
    project="Jet_Impingement_GNN",
    config={
        "learning_rate":    LEARNING_RATE,
        "epochs":           EPOCHS,
        "batch_accumulation": ACCUMULATION_STEPS,
        "weight_decay":     WEIGHT_DECAY,
        "grad_clip":        GRAD_CLIP,
        "device":           str(DEVICE),
        "model":            "JetLatentGNN_v4_signed_dist_features",
        "in_features":      IN_FEATURES,
        "hidden_features":  128,
        "latent_layers":    4,
        "loss_type":        "zscore_pure_data_driven_v2",
        "fixes":            "signed_dist+stag_weight+vel_loss+epochs",
    }
)

# ── Spatial importance weights ─────────────────────
def get_spatial_weight(fine_pos):
    xs, ys, zs = fine_pos[:, 0], fine_pos[:, 1], fine_pos[:, 2]
    radius = torch.sqrt(
        (xs - GEOM['jet_center_x'])**2 + (zs - GEOM['jet_center_z'])**2
    )
    peak_weight = torch.exp(-(radius / GEOM['D_m'])**2) * 10.0
    wall_weight = (ys < 1e-4).float() * 5.0
    w = 1.0 + peak_weight + wall_weight
    return w


def get_velocity_spatial_weight(fine_pos):
    xs, ys, zs = fine_pos[:, 0], fine_pos[:, 1], fine_pos[:, 2]
    radius = torch.sqrt(
        (xs - GEOM['jet_center_x'])**2 + (zs - GEOM['jet_center_z'])**2
    )
    spread_weight = torch.exp(-((radius - 1.5 * GEOM['D_m']) / GEOM['D_m'])**2) * 12.0
    wall_weight   = (ys < 1e-4).float() * 6.0
    # Fix 2: stagnation weight 3.0 → 15.0
    # The model was capping velocity at ~4 m/s (inlet velocity) because the
    # stagnation zone (radius ≈ 0) had too little weight in the loss.
    # Boosting this forces the model to correctly learn the jet acceleration zone.
    stag_weight   = torch.exp(-(radius / GEOM['D_m'])**2) * 15.0
    w = 1.0 + spread_weight + wall_weight + stag_weight
    return w


# ─────────────────────────────────────────────────
#  Z-SCORE NORMALIZED LOSS (pure data-driven)
# ─────────────────────────────────────────────────
class NormalizedMSELoss(nn.Module):
    def __init__(self, stats_path: str):
        super().__init__()
        stats = torch.load(stats_path, weights_only=True)
        for key in ['mean_T', 'std_T', 'mean_P', 'std_P',
                    'mean_Vx', 'std_Vx', 'mean_Vy', 'std_Vy', 'mean_Vz', 'std_Vz']:
            self.register_buffer(key, torch.tensor(stats[key], dtype=torch.float))

    def _z(self, x, mean, std):
        return (x - mean) / std

    def forward(self, pred, target, spatial_w, vel_spatial_w=None):
        p_T  = self._z(pred[:, 0], self.mean_T,  self.std_T)
        p_P  = self._z(pred[:, 1], self.mean_P,  self.std_P)
        p_Vx = self._z(pred[:, 2], self.mean_Vx, self.std_Vx)
        p_Vy = self._z(pred[:, 3], self.mean_Vy, self.std_Vy)
        p_Vz = self._z(pred[:, 4], self.mean_Vz, self.std_Vz)

        t_T  = self._z(target[:, 0], self.mean_T,  self.std_T)
        t_P  = self._z(target[:, 1], self.mean_P,  self.std_P)
        t_Vx = self._z(target[:, 2], self.mean_Vx, self.std_Vx)
        t_Vy = self._z(target[:, 3], self.mean_Vy, self.std_Vy)
        t_Vz = self._z(target[:, 4], self.mean_Vz, self.std_Vz)

        e_T  = (p_T  - t_T) **2
        e_P  = (p_P  - t_P) **2
        e_Vx = (p_Vx - t_Vx)**2
        e_Vy = (p_Vy - t_Vy)**2
        e_Vz = (p_Vz - t_Vz)**2

        temp_mag = torch.sqrt(target[:, 0].detach().clamp(min=0.0) + 1.0)
        v_w = vel_spatial_w if vel_spatial_w is not None else spatial_w

        loss_T  = (e_T  * spatial_w * temp_mag).mean()
        loss_P  = (e_P  * spatial_w).mean()
        # Fix 3: velocity loss weights 1.5 → 3.0
        # Doubles the penalty for getting velocity wrong relative to pressure.
        # Necessary because U/V/W were converging to near-zero predictions.
        loss_Vx = (e_Vx * v_w * 3.0).mean()
        loss_Vy = (e_Vy * v_w * 3.0).mean()
        loss_Vz = (e_Vz * v_w * 3.0).mean()

        total = loss_T + loss_P + loss_Vx + loss_Vy + loss_Vz
        return total, loss_T.item(), loss_P.item(), loss_Vx.item(), loss_Vy.item(), loss_Vz.item()


# ─────────────────────────────────────────────────
#  FEATURE BUILDER (with signed distance features)
# ─────────────────────────────────────────────────
def build_features(sim_ds, device):
    """
    Builds the 20-feature input tensor for a simulation.
    Called inside the training loop to support signed distance features
    that depend on the graph node positions.

    Feature layout (20 total):
        [0:4]  Global:  Log_Re, Heat_Flux, q_star, temp_grad_proxy
        [4:12] Spatial: X, Y, Z, radius, inv_radius, dist_outflow, dist_outlet, signed_dist_wall
        [12]   Skewed:  y_norm
        [13:18] BC:     stagnation_flag, is_HD4, is_HD6, re_regime, bc_velocity
        [18:20] Dir:    signed_dist_x, signed_dist_z     ← Fix 1
    """
    # sim_ds.X already contains the 18-feature tensor from FeatureEngineering.
    # We append signed_dist_x and signed_dist_z from the graph positions.
    x_in = sim_ds.X.squeeze(0).to(device)          # (N, 18)

    # Get node coordinates from the graph stored in sim_ds (adjust attribute name if needed)
    if hasattr(sim_ds, 'pos'):
        pos = sim_ds.pos.to(device)
    else:
        # Fallback: read from the pre-stored position in X columns 4 and 6
        # (X coord is column 4, Z coord is column 6 in spatial block)
        pos_x = x_in[:, 4]
        pos_z = x_in[:, 6]
        signed_dist_x = (pos_x - GEOM['jet_center_x']).unsqueeze(1)
        signed_dist_z = (pos_z - GEOM['jet_center_z']).unsqueeze(1)
        return torch.cat([x_in, signed_dist_x, signed_dist_z], dim=1)

    signed_dist_x = (pos[:, 0] - GEOM['jet_center_x']).unsqueeze(1)
    signed_dist_z = (pos[:, 2] - GEOM['jet_center_z']).unsqueeze(1)
    return torch.cat([x_in, signed_dist_x, signed_dist_z], dim=1)   # (N, 20)


# ─────────────────────────────────────────────────
#  1. DISCOVER FILES
# ─────────────────────────────────────────────────
train_sim_dir   = os.path.join(DATA_DIR, 'train_sims')
val_sim_dir     = os.path.join(DATA_DIR, 'val_sims')
train_sim_files = sorted(glob.glob(os.path.join(train_sim_dir, '*.pt')))
val_sim_files   = sorted(glob.glob(os.path.join(val_sim_dir,   '*.pt')))

print(f"Found {len(train_sim_files)} train sims, {len(val_sim_files)} val sims.")

# ─────────────────────────────────────────────────
#  2. PRE-LOAD GRAPH TEMPLATES
# ─────────────────────────────────────────────────
print("Pre-loading Graph Templates (HD 4, 5, 6)...")
templates = {}
for hd in [4, 5, 6]:
    path = os.path.join(DATA_DIR, f'processed_graph_sim_HD{hd}_001.pt')
    if os.path.exists(path):
        graph = torch.load(path, weights_only=False).to(DEVICE)
        templates[hd] = graph
        print(f"  HD={hd}: {graph['fine'].pos.shape[0]:,} fine nodes, "
              f"{graph['latent'].pos.shape[0]:,} latent nodes")

# ─────────────────────────────────────────────────
#  3. MODEL, LOSS, OPTIMIZER
# ─────────────────────────────────────────────────
# Fix 1: in_features=20 (was 18) — extra features: signed_dist_x, signed_dist_z
model = JetLatentGNN(
    in_features=IN_FEATURES, hidden_features=128, out_features=5, latent_layers=4
).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {total_params:,}  (in_features={IN_FEATURES})")

stats_path = os.path.join(DATA_DIR, 'training_stats.pt')
criterion  = NormalizedMSELoss(stats_path).to(DEVICE)

optimizer  = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
# Fix 4: eta_min 1e-6 → 5e-5
# Original eta_min=1e-6 caused the LR to decay to near-zero by epoch 100,
# starving velocity learning in the second half of training.
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=5e-5
)

# ─────────────────────────────────────────────────
#  4. TRAINING LOOP
# ─────────────────────────────────────────────────
print("\n" + "="*60)
print(f"  TRAINING — {EPOCHS} epochs, in_features={IN_FEATURES}")
print("  Fixes: signed_dist | stag_weight×15 | vel_loss×3 | eta_min=5e-5")
print("="*60)

best_val_loss = float('inf')
best_path     = os.path.join(DATA_DIR, "best_jet_surrogate_model.pth")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss = 0.0
    n_nan_skips = 0
    optimizer.zero_grad()

    random.shuffle(train_sim_files)
    loop = tqdm(train_sim_files, desc=f"Epoch {epoch:03d}/{EPOCHS} [Train]")

    for i, sim_path in enumerate(loop):
        sim_ds = torch.load(sim_path, weights_only=False)
        t_in   = sim_ds.T.squeeze(0).to(DEVICE)

        assert sim_ds.X.shape[-1] in (18, 20), \
            f"Unexpected feature count {sim_ds.X.shape[-1]} in {sim_path}"

        # Build 20-feature input (appends signed dist if sim_ds has 18 features)
        if sim_ds.X.shape[-1] == 18:
            x_in = build_features(sim_ds, DEVICE)
        else:
            x_in = sim_ds.X.squeeze(0).to(DEVICE)
        del sim_ds

        assert x_in.shape[1] == IN_FEATURES, \
            f"Expected {IN_FEATURES} features, got {x_in.shape[1]}"

        hd_match = re.search(r'H(\d+)', os.path.basename(sim_path))
        hd_val   = int(hd_match.group(1)) if hd_match else 4

        graph = templates[hd_val]
        graph['fine'].x = x_in
        graph['fine'].y = t_in

        pred = model(graph)
        fine_pos      = graph['fine'].pos[:x_in.shape[0]]
        spatial_w     = get_spatial_weight(fine_pos)
        vel_spatial_w = get_velocity_spatial_weight(fine_pos)

        loss, lt, lp, lvx, lvy, lvz = criterion(pred, t_in, spatial_w, vel_spatial_w)

        if torch.isnan(loss):
            n_nan_skips += 1
            continue

        (loss / ACCUMULATION_STEPS).backward()

        if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_sim_files):
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()

        total_train_loss += loss.item()
        loop.set_postfix({
            'T':  f"{lt:.3f}",
            'P':  f"{lp:.3f}",
            'Vx': f"{lvx:.3f}",
            'Vy': f"{lvy:.3f}",
            'Vz': f"{lvz:.3f}",
        })

    avg_train = total_train_loss / max(len(train_sim_files) - n_nan_skips, 1)

    # ── Validation ────────────────────────────────────────────────────────────
    model.eval()
    total_val  = 0.0
    val_t = val_p = val_vx = val_vy = val_vz = 0.0

    with torch.no_grad():
        vbar = tqdm(val_sim_files, desc=f"Epoch {epoch:03d}/{EPOCHS} [Val  ]")
        for sim_path in vbar:
            sim_ds = torch.load(sim_path, weights_only=False)
            t_in   = sim_ds.T.squeeze(0).to(DEVICE)

            if sim_ds.X.shape[-1] == 18:
                x_in = build_features(sim_ds, DEVICE)
            else:
                x_in = sim_ds.X.squeeze(0).to(DEVICE)
            del sim_ds

            hd_match = re.search(r'H(\d+)', os.path.basename(sim_path))
            hd_val   = int(hd_match.group(1)) if hd_match else 4

            graph = templates[hd_val]
            graph['fine'].x = x_in

            pred          = model(graph)
            fine_pos      = graph['fine'].pos
            spatial_w     = get_spatial_weight(fine_pos)
            vel_spatial_w = get_velocity_spatial_weight(fine_pos)

            loss, lt, lp, lvx, lvy, lvz = criterion(
                pred, t_in, spatial_w, vel_spatial_w
            )
            if torch.isnan(loss):
                continue

            total_val += loss.item()
            val_t  += lt
            val_p  += lp
            val_vx += lvx
            val_vy += lvy
            val_vz += lvz

    n_val   = max(len(val_sim_files), 1)
    avg_val = total_val / n_val
    scheduler.step()

    print(
        f"Epoch {epoch:03d} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
        f"LR: {optimizer.param_groups[0]['lr']:.2e}"
    )
    print(
        f"  Val → T:{val_t/n_val:.4f}  P:{val_p/n_val:.4f}  "
        f"Vx:{val_vx/n_val:.4f}  Vy:{val_vy/n_val:.4f}  Vz:{val_vz/n_val:.4f}"
    )

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'in_features':          IN_FEATURES,   # saved for reference in inference
            'val_loss':             best_val_loss,
        }, best_path)
        print(f"  Best model saved → {best_path} (val={best_val_loss:.4f})")

    wandb.log({
        "epoch/train_loss": avg_train,
        "epoch/val_loss":   avg_val,
        "epoch/val_T":      val_t  / n_val,
        "epoch/val_P":      val_p  / n_val,
        "epoch/val_Vx":     val_vx / n_val,
        "epoch/val_Vy":     val_vy / n_val,
        "epoch/val_Vz":     val_vz / n_val,
        "epoch/lr":         optimizer.param_groups[0]['lr'],
        "epoch":            epoch,
    })

wandb.finish()
print("\nTraining complete.")
print(f"Best val loss: {best_val_loss:.4f}  |  Checkpoint: {best_path}")