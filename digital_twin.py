import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# 1. THE DATA LOADER & GLOBAL NORMALIZER
# ==============================================================================
class CFDTensorDataset(Dataset):
    def __init__(self, npz_dir):
        """Loads the .npz files and applies safe global normalization."""
        self.files = glob.glob(os.path.join(npz_dir, "*.npz"))
        
        # ⚠️ IMPORTANT: These are placeholder global bounds. 
        # You will need to calculate the actual absolute Min/Max for your 100 trials.
        # Format: [Min, Max]
        self.global_bounds = {
            0: [20.0000, 61.7186],  # Temperature
            1: [-75042.7812, 452114.2500],  # Pressure
            2: [0.0000, 37.2057],  # Turbulent_Kinetic_Energy
            3: [-25.4716, 25.3899],  # Velocity_X
            4: [-10.3894, 10.6345],  # Velocity_Y
            5: [-9.1371, 9.0707]  # Velocity_Z
        }

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load the highly compressed tensor file
        data = np.load(self.files[idx])
        inputs = data['inputs']   # Shape: (3, 128, 128, 128)
        targets = data['targets'] # Shape: (6, 128, 128, 128)
        
        # Apply Global Min-Max Scaling to the TARGETS (0 to 1 range)
        for c in range(6):
            c_min, c_max = self.global_bounds[c]
            targets[c] = (targets[c] - c_min) / (c_max - c_min)
            
        # Re-apply the fluid mask to the targets to ensure empty space stays perfectly zero
        mask = inputs[0]
        targets = targets * mask
        
        # Convert to PyTorch tensors
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

# ==============================================================================
# 2. THE 3D U-NET ARCHITECTURE
# ==============================================================================
class DoubleConv3D(nn.Module):
    """(Conv3D -> BatchNorm -> GELU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Note: We use GELU instead of ReLU because fluid dynamics have smooth 
        # gradients. ReLU's sharp cutoff can cause blocky predictions in physics.
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.double_conv(x)


class CFDUnet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=6):
        super().__init__()
        
        # ENCODER (Downsampling)
        # 128x128x128
        self.inc = DoubleConv3D(in_channels, 16)
        # 64x64x64
        self.down1 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(16, 32))
        # 32x32x32
        self.down2 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(32, 64))
        # 16x16x16
        self.down3 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(64, 128))
        
        # BOTTLENECK (The deep physics representation)
        # 8x8x8
        self.down4 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(128, 256))

        # DECODER (Upsampling & Concatenation)
        self.up1 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv3D(256, 128)
        
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv3D(128, 64)
        
        self.up3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv3D(64, 32)
        
        self.up4 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv3D(32, 16)

        # FINAL OUTPUT MAPPER
        # Maps the 16 features down to your 6 physics channels
        self.outc = nn.Conv3d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # Extract original mask so we can enforce boundary conditions at the end
        fluid_mask = x[:, 0:1, :, :, :] 
        
        # Encode
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) # Bottleneck

        # Decode (with Skip Connections to preserve sharp geometry)
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up3(x)
        
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up4(x)

        # Output predictions
        predictions = self.outc(x)
        
        # CRITICAL PHYSICS CONSTRAINT: 
        # Multiply by the fluid mask so predictions inside solid walls are exactly 0.0
        return predictions * fluid_mask

# ==============================================================================
# 3. QUICK TEST (Ensuring the plumbing works)
# ==============================================================================
if __name__ == "__main__":
    print("Initializing 3D U-Net...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    
    model = CFDUnet3D().to(device)
    
    # Create a dummy batch of 1 trial (Batch Size, Channels, X, Y, Z)
    print("Testing forward pass with dummy 128x128x128 tensor...")
    dummy_input = torch.randn(1, 3, 128, 128, 128).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    print(f"Success! Model output shape: {output.shape}")
    print("Expected: torch.Size([1, 6, 128, 128, 128])")