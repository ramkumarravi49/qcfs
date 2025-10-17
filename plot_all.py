import os
import pandas as pd
import matplotlib.pyplot as plt

# Load results CSV
df = pd.read_csv("qcfs_inference_results.csv")

# Ensure plots folder exists
os.makedirs("plots", exist_ok=True)

# Models in the CSV
models = df['model'].unique()

# --- 1. Accuracy vs Timesteps ---
for model in models:
    subset = df[df['model'] == model]
    plt.figure(figsize=(6,4))
    for L in sorted(subset['L'].unique()):
        subL = subset[subset['L'] == L]
        plt.plot(subL['T'], subL['accuracy'], marker='o', label=f"L={L}")
    plt.xscale('log', base=2)
    plt.xlabel("Timesteps (T)")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{model.upper()} Accuracy vs Timesteps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{model}_accuracy_vs_T.png")
    plt.close()

# --- 2. Spikes vs Timesteps ---
for model in models:
    subset = df[df['model'] == model]
    plt.figure(figsize=(6,4))
    for L in sorted(subset['L'].unique()):
        subL = subset[subset['L'] == L]
        plt.plot(subL['T'], subL['avg_spikes'], marker='s', label=f"L={L}")
    plt.xscale('log', base=2)
    plt.xlabel("Timesteps (T)")
    plt.ylabel("Average spikes per inference")
    plt.title(f"{model.upper()} Spikes vs Timesteps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{model}_spikes_vs_T.png")
    plt.close()

# --- 3. Accuracy vs Spikes (Energy–Accuracy tradeoff) ---
for model in models:
    subset = df[df['model'] == model]
    plt.figure(figsize=(6,4))
    for L in sorted(subset['L'].unique()):
        subL = subset[subset['L'] == L]
        plt.plot(subL['avg_spikes'], subL['accuracy'], marker='d', label=f"L={L}")
    plt.xlabel("Average spikes per inference")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{model.upper()} Accuracy vs Spikes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{model}_accuracy_vs_spikes.png")
    plt.close()

# --- 4. Bar chart at small T (T=2 and T=4) ---
for model in models:
    subset = df[df['model'] == model]
    for smallT in [2,4]:
        subT = subset[subset['T'] == smallT]
        plt.figure(figsize=(6,4))
        plt.bar(subT['L'].astype(str), subT['accuracy'])
        plt.xlabel("Quantization step (L)")
        plt.ylabel("Accuracy (%)")
        plt.title(f"{model.upper()} Accuracy at T={smallT}")
        plt.tight_layout()
        plt.savefig(f"plots/{model}_bar_T{smallT}.png")
        plt.close()

print("✅ All plots saved in ./plots/")
