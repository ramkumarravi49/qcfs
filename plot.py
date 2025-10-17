# import matplotlib.pyplot as plt
# import numpy as np

# # Time‐steps
# time_steps = np.array([1, 2, 4, 8, 16, 32, 64, 128])

# # VGG-16 accuracies on CIFAR-10
# acc_no_reg = np.array([62.89, 83.93, 91.77, 94.45, 95.22, 95.56, 95.74, 95.79]) / 100.0
# acc_reg    = np.array([71.48, 86.30, 93.73, 95.25, 95.63, 95.78, 95.88, 95.85]) / 100.0

# # Use a built-in style
# plt.style.use('ggplot')

# fig, ax = plt.subplots(figsize=(6, 4))

# # Plot the two curves
# ax.plot(time_steps, acc_no_reg, marker='o', linestyle='-', color='tab:green', label='w_q = 0 (no reg)')
# ax.plot(time_steps, acc_reg,    marker='s', linestyle='-', color='tab:blue',  label='w_q = 0.2 (with reg)')

# # Log2 x-axis
# ax.set_xscale('log', base=2)
# ax.set_xticks(time_steps)
# ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

# ax.set_xlabel('Simulation time-steps')
# ax.set_ylabel('Accuracy')
# ax.set_title('VGG-16 on CIFAR-10: QCFS Regularizer Effect')
# ax.legend(loc='lower right')

# plt.tight_layout()
# plt.savefig("qcfs_effect.png")
# plt.show()



import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV
df = pd.read_csv("qcfs_results_clean.csv")  # with columns: model,L,T,accuracy,avg_spikes

# Accuracy vs Timesteps
plt.figure(figsize=(6,4))
for model in df['model'].unique():
    subset = df[df['model'] == model]
    plt.plot(subset['T'], subset['accuracy'], marker='o', label=model)
plt.xscale('log', base=2)
plt.xlabel("Timesteps (T)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Timesteps")
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_vs_timesteps.png")

# Avg Spikes vs Timesteps
plt.figure(figsize=(6,4))
for model in df['model'].unique():
    subset = df[df['model'] == model]
    plt.plot(subset['T'], subset['avg_spikes'], marker='s', label=model)
plt.xscale('log', base=2)
plt.xlabel("Timesteps (T)")
plt.ylabel("Avg spikes per inference")
plt.title("Spikes vs Timesteps")
plt.legend()
plt.tight_layout()
plt.savefig("spikes_vs_timesteps.png")

# Accuracy vs Avg Spikes (Energy–Accuracy tradeoff)
plt.figure(figsize=(6,4))
for model in df['model'].unique():
    subset = df[df['model'] == model]
    plt.plot(subset['avg_spikes'], subset['accuracy'], marker='d', label=model)
plt.xlabel("Avg spikes per inference")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Spikes (Energy–Accuracy)")
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_vs_spikes.png")

plt.show()
