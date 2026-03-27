import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
CSV_PATH = r"saved_models\training_history.csv"

if not os.path.exists(CSV_PATH):
    print("❌ Cannot find training_history.csv! Make sure the path is correct.")
    exit()

print("Loading training history...")
df = pd.read_csv(CSV_PATH)

# --- PLOTTING ---
# We create a figure with 2 subplots (one for global loss, one for physics breakdown)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- PLOT 1: Global Train vs Val Loss ---
ax1.plot(df["Epoch"], df["Train_Loss"], label="Training Loss", color="blue", alpha=0.7, linewidth=2)
ax1.plot(df["Epoch"], df["Val_Total"], label="Validation Loss", color="red", alpha=0.9, linewidth=2)

ax1.set_title("Global U-Net Convergence")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Smooth L1 Loss (Log Scale)")
ax1.set_yscale('log') # Log scale is crucial for seeing the tiny drops at the end
ax1.grid(True, which="both", ls="--", alpha=0.5)
ax1.legend()

# --- PLOT 2: The Physics Battleground (Val Only) ---
ax2.plot(df["Epoch"], df["Val_Temp"], label="Temperature", color="darkorange", linewidth=2)
ax2.plot(df["Epoch"], df["Val_Press"], label="Pressure", color="purple", linewidth=2, alpha=0.8)
ax2.plot(df["Epoch"], df["Val_Velocity"], label="Velocity", color="teal", linewidth=2)

ax2.set_title("Physics-Specific Convergence")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Component Error (Log Scale)")
ax2.set_yscale('log')
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.show()