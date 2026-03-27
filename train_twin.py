import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm  
from digital_twin import CFDTensorDataset, CFDUnet3D

# --- 1. CONFIGURATION ---
TENSOR_DIR = r"C:\Users\sudee\Desktop\work\cfd\ML_Tensors"
BATCH_SIZE = 2  
EPOCHS = 500
LEARNING_RATE = 1e-4

# --- 2. SETUP DATASET & DATALOADER ---
print("Loading dataset...")
full_dataset = CFDTensorDataset(TENSOR_DIR)

train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

print(f"Dataset Split: {train_size} Training | {val_size} Validation")

# --- 3. INITIALIZE MODEL, LOSS, OPTIMIZER, & SCHEDULER ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training on device: {device}")

model = CFDUnet3D().to(device)

criterion = nn.SmoothL1Loss() 
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# 🔥 UPGRADE: Learning Rate Scheduler
# If the validation loss stops improving for 20 epochs, cut the learning rate in half.
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

scaler = torch.cuda.amp.GradScaler()

best_val_loss = float('inf')
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

# 🔥 UPGRADE: CSV Logger Setup
csv_path = os.path.join(save_dir, "training_history.csv")
with open(csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    # Write the header
    writer.writerow(["Epoch", "LR", "Train_Loss", "Val_Total", "Val_Temp", "Val_Press", "Val_Velocity"])

print("\n" + "="*70)
print("🔥 BEGINNING DIGITAL TWIN TRAINING (WITH DEEP ANALYSIS)")
print("="*70)

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_train_loss = 0.0
    
    train_loop = tqdm(train_loader, desc=f"Epoch [{epoch:03d}/{EPOCHS}] Train", leave=False)
    
    for inputs, targets in train_loop:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_train_loss += loss.item()
        train_loop.set_postfix(loss=loss.item())
        
    avg_train_loss = epoch_train_loss / len(train_loader)
    
    # --- VALIDATION PHASE & DEEP ANALYSIS ---
    model.eval()
    epoch_val_loss = 0.0
    
    # Physics Trackers
    val_temp_loss = 0.0
    val_press_loss = 0.0
    val_vel_loss = 0.0
    
    val_loop = tqdm(val_loader, desc=f"Epoch [{epoch:03d}/{EPOCHS}] Val  ", leave=False)
    
    with torch.no_grad():
        for inputs, targets in val_loop:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.cuda.amp.autocast():
                predictions = model(inputs)
                
                # Global Loss (for the optimizer)
                val_loss = criterion(predictions, targets)
                epoch_val_loss += val_loss.item()
                
                # 🔥 UPGRADE: Per-Channel Physics Loss Analysis
                # Channel 0: Temp | Channel 1: Pressure | Channels 3,4,5: Vx, Vy, Vz
                val_temp_loss += criterion(predictions[:, 0], targets[:, 0]).item()
                val_press_loss += criterion(predictions[:, 1], targets[:, 1]).item()
                val_vel_loss += criterion(predictions[:, 3:6], targets[:, 3:6]).item()
                
            val_loop.set_postfix(loss=val_loss.item())
            
    # Calculate Averages
    avg_val_loss = epoch_val_loss / len(val_loader)
    avg_temp_loss = val_temp_loss / len(val_loader)
    avg_press_loss = val_press_loss / len(val_loader)
    avg_vel_loss = val_vel_loss / len(val_loader)
    
    # Step the scheduler based on the global validation loss
    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # --- TERMINAL OUTPUT ---
    print(f"Epoch [{epoch:03d}/{EPOCHS}] | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f} | "
          f"Temp: {avg_temp_loss:.5f} | Press: {avg_press_loss:.5f} | Vel: {avg_vel_loss:.5f}")
    
    # --- LOG TO CSV ---
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, current_lr, avg_train_loss, avg_val_loss, avg_temp_loss, avg_press_loss, avg_vel_loss])
    
    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(save_dir, "best_digital_twin.pth"))
        print("   🌟 New best model saved!")

print("\n🎉 FULL TRAINING RUN COMPLETE!")