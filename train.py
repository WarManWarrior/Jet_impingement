"""
=============================================================================
  TRAINING LOOP — JET IMPINGEMENT LATENT GNN SURROGATE

  CHANGES FROM PREVIOUS VERSION (ported from working v4 code):
    NEW  Z-score normalized loss (NormalizedMSELoss):
         Ported from v4 NormalizedMSELoss. Normalizes predictions AND targets
         to zero-mean unit-variance before computing MSE. Uses per-channel
         mean and std from training_stats.pt (saved by FeatureEngineering.py).
         This makes all 5 channels (T, P, Vx, Vy, Vz) contribute equal gradient
         weight, replacing the brittle manual feature_weights tensor. The v4
         code comment explicitly calls out a "~2×10⁹ imbalance" between T and P.

    NEW  Two-phase training (ported from v4):
         Phase 1 (epochs 1–PHASE2_EPOCH): Z-score MSE on all 5 channels.
         Phase 2 (epochs PHASE2_EPOCH+1 → end): adds outlet pressure BC loss.
         The BC loss penalizes outlet node predicted pressures deviating from
         the Z-score of 0 Pa (gauge pressure at outlets). This pins the pressure
         field once the model has learned the overall structure in Phase 1.

    NEW  Outlet pressure BC loss:
         Ported from v4. Detects outlet nodes from graph['fine'].pos coordinates
         (x < EPS or x > x_max - EPS, near z=jet_center_z, low y).
         BC target is loaded from training_stats.pt['outlet_P_bc_target'] which
         is the Z-score of 0 Pa in scaled log-pressure space.

    NEW  CosineAnnealingLR:
         Replaces ReduceLROnPlateau. Smooth 150-epoch schedule prevents the
         aggressive early LR decay that was stopping gradient flow before the
         model converged on hot nodes.

    FIX  in_features updated to 15 for new bc_velocity feature.
    FIX  Thermal magnitude weight retained (prevents ambient collapse).
         No manual P/T multipliers — Z-score handles channel balancing.
         FIX: in_features updated to 16 for signed_dist_wall + Parameter Node
=============================================================================
"""

import os
import re
import glob
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np

from FeatureEngineering import JetImpingementDataset, STAGNATION_FLAG_IDX, GEOM
from LatentGNN import JetLatentGNN

# ─────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────
DATA_DIR           = r"D:\data\JET"
EPOCHS             = 150    # extended to match v4
ACCUMULATION_STEPS = 8
LEARNING_RATE      = 3e-4   # v4 value
WEIGHT_DECAY       = 1e-4
GRAD_CLIP          = 1.0
DEVICE             = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Two-phase training (ported from v4)
PHASE2_EPOCH   = 80    # Phase 2 (outlet BC loss) starts at epoch 81
W_BC           = 10.0  # weight for outlet pressure BC loss in Phase 2
PRESSURE_BOOST = 2.0   # extra P weight in Phase 2 (lighter than v4's 5.0 — our log-P is already compressed)

# Outlet detection geometry (matches FeatureEngineering constants)
D_M         = GEOM['D_m']
JET_CENTER_Z = GEOM['jet_center_z']
X_MAX       = GEOM['domain_x_max']
EPS         = 1e-4

print(f"Using device: {DEVICE}")

# ── NEW: Spatial importance weight (replacing v1 stagnation flag logic) ──
def get_spatial_weight(fine_pos):
    """
    Computes per-node importance weights based on physical location.
    Stagnation Zone: Peak variance, 8x weight.
    Wall boundary: Critical BL physics, 4x weight.
    """
    xs, ys, zs = fine_pos[:,0], fine_pos[:,1], fine_pos[:,2]
    # Stagnation zone: r < D and near chip surface (y < 1mm)
    stagnation = ((xs - GEOM['jet_center_x'])**2 + (zs - GEOM['jet_center_z'])**2 < (GEOM['D_m'])**2) & (ys < 0.001)
    # Wall: direct surface interaction
    wall = ys < 1e-4
    
    w = torch.ones(len(xs), device=DEVICE)
    w[stagnation] = 8.0
    w[wall] *= 4.0  # compounding if overlapping
    return w


# ─────────────────────────────────────────────────
#  Z-SCORE NORMALIZED LOSS   (ported from v4 NormalizedMSELoss)
# ─────────────────────────────────────────────────
class NormalizedMSELoss(nn.Module):
    """
    Z-score normalizes predictions and targets before computing MSE.
    This makes all 5 output channels contribute equal gradient weight,
    fixing the scale imbalance between temperature (°C) and pressure (Pa).

    Per-channel stats are loaded from training_stats.pt which is written
    by FeatureEngineering.py fit_and_transform() using the actual scaled
    training targets — not assumed to be (0, 1).

    outlet_P_bc_target is the Z-score of 0 Pa (gauge) in scaled log-pressure
    space. When the model predicts this value at outlet nodes, it is predicting
    exactly 0 Pa gauge pressure, which is the physical outlet BC.
    """
    def __init__(self, stats_path: str):
        super().__init__()
        stats = torch.load(stats_path, weights_only=True)

        for key in ['mean_T', 'std_T', 'mean_P', 'std_P',
                    'mean_Vx', 'std_Vx', 'mean_Vy', 'std_Vy', 'mean_Vz', 'std_Vz',
                    'outlet_P_bc_target']:
            self.register_buffer(key, torch.tensor(stats[key], dtype=torch.float))

        print(f"[Loss]  Outlet BC target (Z-score of 0 Pa) = {stats['outlet_P_bc_target']:.4f}")
        print(f"[Loss]  mean_P_log={stats['mean_P']:.4f}, std_P_log={stats['std_P']:.4f}")

    def _z(self, x, mean, std):
        return (x - mean) / std

    def forward(self, pred, target, spatial_w, outlet_mask=None, phase2=False):
        """
        pred         : (N, 5)  — model output in scaled space [T, P, Vx, Vy, Vz]
        target       : (N, 5)  — ground truth in scaled space
        spatial_w    : (N,)    — importance weights (8x stag, 4x wall)
        outlet_mask  : (N,) bool — True at outlet nodes
        phase2       : bool — enables BC loss and pressure boost
        """
        # Z-score normalize all channels
        p_T  = self._z(pred[:,0],   self.mean_T,  self.std_T)
        p_P  = self._z(pred[:,1],   self.mean_P,  self.std_P)
        p_Vx = self._z(pred[:,2],   self.mean_Vx, self.std_Vx)
        p_Vy = self._z(pred[:,3],   self.mean_Vy, self.std_Vy)
        p_Vz = self._z(pred[:,4],   self.mean_Vz, self.std_Vz)

        t_T  = self._z(target[:,0], self.mean_T,  self.std_T)
        t_P  = self._z(target[:,1], self.mean_P,  self.std_P)
        t_Vx = self._z(target[:,2], self.mean_Vx, self.std_Vx)
        t_Vy = self._z(target[:,3], self.mean_Vy, self.std_Vy)
        t_Vz = self._z(target[:,4], self.mean_Vz, self.std_Vz)

        # Per-node squared errors
        e_T  = (p_T  - t_T) **2
        e_P  = (p_P  - t_P) **2
        e_Vx = (p_Vx - t_Vx)**2
        e_Vy = (p_Vy - t_Vy)**2
        e_Vz = (p_Vz - t_Vz)**2

        # Thermal magnitude weight (retained for temperature stability)
        temp_mag = (target[:, 0].detach().clamp(min=0.0) + 0.5)

        loss_T  = (e_T  * spatial_w * temp_mag).mean()
        loss_P  = (e_P  * spatial_w).mean()
        loss_Vx = (e_Vx * spatial_w).mean()
        loss_Vy = (e_Vy * spatial_w).mean()
        loss_Vz = (e_Vz * spatial_w).mean()

        p_weight = PRESSURE_BOOST if phase2 else 1.0
        total = loss_T + p_weight * loss_P + loss_Vx + loss_Vy + loss_Vz

        # Outlet pressure BC loss — Phase 2 only
        loss_bc = torch.tensor(0.0, device=pred.device)
        if phase2 and outlet_mask is not None and outlet_mask.any():
            # p_P[outlet_mask] should equal outlet_P_bc_target (Z-score of 0 Pa)
            loss_bc = ((p_P[outlet_mask] - self.outlet_P_bc_target) ** 2).mean()
            total   = total + W_BC * loss_bc

        return total, loss_T.item(), loss_P.item(), loss_Vx.item(), loss_Vy.item(), loss_Vz.item(), loss_bc.item()


# ─────────────────────────────────────────────────
#  OUTLET NODE DETECTION
# ─────────────────────────────────────────────────
def get_outlet_mask(fine_pos: torch.Tensor) -> torch.Tensor:
    """
    Detects outlet nodes from spatial coordinates.
    Outlets: at x≈0 or x≈x_max boundary, near z=jet_center_z, within outlet radius.
    Ported from v4 outlet_flag logic.
    """
    xs = fine_pos[:, 0]
    zs = fine_pos[:, 2]
    D_outlet = D_M / 2.0
    r_outlet = D_outlet / 2.0
    at_x_boundary = (xs < EPS) | (xs > X_MAX - EPS)
    near_z_center  = torch.abs(zs - JET_CENTER_Z) < r_outlet
    return at_x_boundary & near_z_center


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
outlet_masks = {}   # cached per HD
for hd in [4, 5, 6]:
    path = os.path.join(DATA_DIR, f'processed_graph_sim_HD{hd}_001.pt')
    if os.path.exists(path):
        graph = torch.load(path, weights_only=False).to(DEVICE)
        templates[hd]     = graph
        outlet_masks[hd]  = get_outlet_mask(graph['fine'].pos)
        print(f"  ✓ HD={hd}: {graph['fine'].pos.shape[0]:,} fine nodes, "
              f"{graph['latent'].pos.shape[0]:,} latent nodes, "
              f"{outlet_masks[hd].sum().item():,} outlet nodes")
    else:
        print(f"  ⚠ HD={hd} template not found.")

# ─────────────────────────────────────────────────
#  3. MODEL, LOSS, OPTIMIZER
# ─────────────────────────────────────────────────
model = JetLatentGNN(in_features=16, hidden_features=128, out_features=5, latent_layers=4).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {total_params:,}")

stats_path = os.path.join(DATA_DIR, 'training_stats.pt')
criterion  = NormalizedMSELoss(stats_path).to(DEVICE)

optimizer  = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# CosineAnnealingLR: smooth 150-epoch schedule (replaces ReduceLROnPlateau)
# Prevents aggressive early LR decay that was stopping gradient flow on hot nodes
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)


# ─────────────────────────────────────────────────
#  4. TRAINING LOOP
# ─────────────────────────────────────────────────
print("\n" + "="*60)
print("  INITIATING TRAINING — Z-score loss | Two-phase | Global node")
print("="*60)

best_val_loss = float('inf')
best_path     = os.path.join(DATA_DIR, "best_jet_surrogate_model.pth")

for epoch in range(1, EPOCHS + 1):
    phase2    = epoch > PHASE2_EPOCH
    phase_str = "Phase 2 [T+V+P+BC]" if phase2 else "Phase 1 [T+V+P]"

    model.train()
    total_train_loss = 0.0
    n_nan_skips = 0
    optimizer.zero_grad()

    random.shuffle(train_sim_files)
    loop = tqdm(train_sim_files, desc=f"Epoch {epoch:03d}/{EPOCHS} {phase_str} [Train]")

    for i, sim_path in enumerate(loop):
        sim_ds = torch.load(sim_path, weights_only=False)
        x_in   = sim_ds.X.squeeze(0).to(DEVICE)   # [N, 16]
        t_in   = sim_ds.T.squeeze(0).to(DEVICE)   # [N, 5]
        del sim_ds

        assert x_in.shape[1] == 16, f"Expected 16 features, got {x_in.shape[1]}"

        hd_match = re.search(r'H(\d+)', os.path.basename(sim_path))
        if not hd_match:
            hd_match = re.search(r'[/\\]H(\d+)[/\\]', sim_path)
        hd_val = int(hd_match.group(1)) if hd_match else 4

        graph        = templates[hd_val]
        outlet_mask  = outlet_masks[hd_val]
        graph['fine'].x = x_in
        graph['fine'].y = t_in

        pred = model(graph)
        spatial_w = get_spatial_weight(graph['fine'].pos)

        loss, lt, lp, lvx, lvy, lvz, lbc = criterion(
            pred, t_in,
            spatial_w   = spatial_w,
            outlet_mask = outlet_mask,
            phase2      = phase2,
        )

        if torch.isnan(loss):
            n_nan_skips += 1
            optimizer.zero_grad()
            if os.path.exists(best_path):
                model.load_state_dict(
                    torch.load(best_path, weights_only=True)['model_state_dict']
                )
            continue

        (loss / ACCUMULATION_STEPS).backward()

        if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_sim_files):
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()

        total_train_loss += loss.item()
        loop.set_postfix({
            'T': f"{lt:.3f}", 'P': f"{lp:.3f}",
            'Vx': f"{lvx:.3f}", 'Vy': f"{lvy:.3f}", 'Vz': f"{lvz:.3f}",
            'BC': f"{lbc:.3f}",
        })

    avg_train = total_train_loss / max(len(train_sim_files) - n_nan_skips, 1)

    if str(DEVICE) == 'cuda':
        vram = torch.cuda.memory_allocated(DEVICE) / (1024**3)
        print(f"   ► VRAM: {vram:.2f} GB  | NaN skips: {n_nan_skips}")

    # ── VALIDATION ──────────────────────────────────────────────────
    model.eval()
    total_val = 0.0
    val_t = val_p = val_vx = val_vy = val_vz = val_bc = 0.0

    with torch.no_grad():
        vbar = tqdm(val_sim_files, desc=f"Epoch {epoch:03d}/{EPOCHS} {phase_str} [Val  ]")
        for sim_path in vbar:
            sim_ds = torch.load(sim_path, weights_only=False)
            x_in   = sim_ds.X.squeeze(0).to(DEVICE)
            t_in   = sim_ds.T.squeeze(0).to(DEVICE)
            assert x_in.shape[1] == 16
            del sim_ds

            hd_match = re.search(r'H(\d+)', os.path.basename(sim_path))
            if not hd_match:
                hd_match = re.search(r'[/\\]H(\d+)[/\\]', sim_path)
            hd_val = int(hd_match.group(1)) if hd_match else 4

            graph       = templates[hd_val]
            outlet_mask = outlet_masks[hd_val]
            graph['fine'].x = x_in

            pred = model(graph)
            spatial_w = get_spatial_weight(graph['fine'].pos)
            
            loss, lt, lp, lvx, lvy, lvz, lbc = criterion(
                pred, t_in,
                spatial_w   = spatial_w,
                outlet_mask = outlet_mask,
                phase2      = phase2,
            )
            if torch.isnan(loss):
                continue

            total_val += loss.item()
            val_t  += lt;  val_p  += lp
            val_vx += lvx; val_vy += lvy; val_vz += lvz; val_bc += lbc

    n_val    = max(len(val_sim_files), 1)
    avg_val  = total_val / n_val
    scheduler.step()

    print(f"Epoch {epoch:03d} | {phase_str} | "
          f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"  Val → T:{val_t/n_val:.4f}  P:{val_p/n_val:.4f}  "
          f"Vx:{val_vx/n_val:.4f}  Vy:{val_vy/n_val:.4f}  "
          f"Vz:{val_vz/n_val:.4f}  BC:{val_bc/n_val:.4f}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, best_path)
        print(f"Best model → {best_path}  (val={best_val_loss:.4f})")

print("\nTraining complete.")