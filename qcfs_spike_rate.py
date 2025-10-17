import argparse
import os
import torch
import matplotlib.pyplot as plt
from Models import modelpool
from Preprocess import datapool
from Models.layer import IF  # QCFS uses IF neurons

device = "cuda" if torch.cuda.is_available() else "cpu"

# def test_with_spikes(model, test_loader, device):
#     model.eval()
#     correct, total = 0, 0
#     number_of_neurons, spike_sum_over_samples = [], []
#     layer_index, hooks = {}, []
#     total_samples, current_batch_size = 0, 0

#     def hook_fn(module, inputs, out):
#         nonlocal current_batch_size
#         spk = (out > 0).float()  # binary spikes
#         if spk.size(0) == current_batch_size:
#             spk = spk.transpose(0, 1)  # QCFS outputs [T, B, ...]

#         # Flatten spatial dims
#         feature_dims = tuple(range(2, spk.dim()))
#         num_neurons = 1
#         for d in feature_dims: num_neurons *= spk.size(d)

#         spikes_per_sample = spk.sum(dim=(0,) + feature_dims) / num_neurons
#         idx = layer_index.get(module)
#         if idx is None:
#             idx = len(number_of_neurons)
#             layer_index[module] = idx
#             number_of_neurons.append(num_neurons)
#             spike_sum_over_samples.append(spikes_per_sample.sum().item())
#         else:
#             spike_sum_over_samples[idx] += spikes_per_sample.sum().item()

#     # Hook every IF neuron
#     for m in model.modules():
#         if isinstance(m, IF):
#             hooks.append(m.register_forward_hook(hook_fn))

#     with torch.no_grad():
#         for inputs, targets in test_loader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             current_batch_size = inputs.size(0)
#             total_samples += current_batch_size
#             outputs = model(inputs)

#             if outputs.dim() == 3:  # QCFS: [T, B, C]
#                 mean_out = outputs.mean(0)  # average over T -> [B, C]
#             else:
#                 mean_out = outputs

#             _, predicted = mean_out.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()

#     for h in hooks:
#         h.remove()

#     number_of_spikes = [s / total_samples for s in spike_sum_over_samples]
#     acc = 100.0 * correct / total

#     print("number_of_neurons:", number_of_neurons)
#     print("number_of_spikes:", number_of_spikes)
#     print(f"Test Accuracy: {acc:.2f}%")

#     return acc, number_of_neurons, number_of_spikes

def test_with_spikes(model, test_loader, device):
    model.eval()
    correct, total = 0, 0

    number_of_neurons = []
    spike_sum_over_samples = []
    layer_index = {}
    hooks = []
    module_call_counter = {}

    current_batch_size = 0
    total_samples = 0

    # Map module object -> name (for readable keys)
    module_to_name = {m: n for n, m in model.named_modules()}

    # old  hook
    def hook_fn(module, inputs, out):
        nonlocal current_batch_size

        # Binary spike events (QCFS IF gives 0/θ, so thresholding is correct here)
        spk = (out > 0).float()

        # Handle time dimension safely
        if spk.dim() >= 3 and spk.size(0) == current_batch_size:
            # If output is [B, T, ...], make it [T, B, ...]
            spk = spk.transpose(0, 1)

        # Flatten feature dims
        feature_dims = tuple(range(2, spk.dim()))
        num_neurons = 1
        for d in feature_dims:
            num_neurons *= spk.size(d)

        # Average spikes per sample
        spikes_per_sample = spk.sum(dim=(0,) + feature_dims) / num_neurons

        # Track how many times this module is called per forward
        cnt = module_call_counter.get(module, 0) + 1
        module_call_counter[module] = cnt

        mod_name = module_to_name.get(module, str(module))
        key = (mod_name, cnt)

        idx = layer_index.get(key)
        if idx is None:
            idx = len(number_of_neurons)
            layer_index[key] = idx
            number_of_neurons.append(num_neurons)
            spike_sum_over_samples.append(spikes_per_sample.sum().item())
        else:
            spike_sum_over_samples[idx] += spikes_per_sample.sum().item()

    # new hook 
    def hook_fn(module, inputs, out):
        nonlocal current_batch_size

        # Convert to binary events first (IF outputs are 0 or theta)
        spk = (out > 0).float()

        # ---- Robust handling of possible layouts ----
        # Cases we might see for 'spk':
        # 1) [T, B, ...]          -> good
        # 2) [B, T, ...]          -> transpose to [T, B, ...]
        # 3) [T*B, C, H, W, ...]  -> produced by merge(flatten(0,1)); reshape to [T, B, ...]
        if spk.dim() >= 3:
            # Case 2: batch-first but time is second: detect by first-dim == batch
            if spk.size(0) == current_batch_size:
                # [B, T, ...] -> [T, B, ...]
                spk = spk.transpose(0, 1)
            else:
                # Case 3 detection: first-dim == T * batch  (module.T exists on IF)
                mod_T = getattr(module, "T", None)
                if mod_T is not None and mod_T > 0 and spk.size(0) == current_batch_size * int(mod_T):
                    # reshape back to [T, B, ...]
                    new_shape = (int(mod_T), current_batch_size) + tuple(spk.shape[1:])
                    spk = spk.view(new_shape)
                # else: if none of the above, leave as-is (fallback)

        # Now spk should be [T, B, ...] (or at least have time dim at 0)
        feature_dims = tuple(range(2, spk.dim()))
        num_neurons = 1
        for d in feature_dims:
            num_neurons *= spk.size(d)

        # spikes_per_sample is a vector of length B: total spikes across TIME and spatial dims, normalized by #neurons
        spikes_per_sample = spk.sum(dim=(0,) + feature_dims) / num_neurons

        # rest of accumulation follows TET-style (module call counting to separate repeated modules)
        cnt = module_call_counter.get(module, 0) + 1
        module_call_counter[module] = cnt
        mod_name = module_to_name.get(module, str(module))
        key = (mod_name, cnt)

        idx = layer_index.get(key)
        if idx is None:
            idx = len(number_of_neurons)
            layer_index[key] = idx
            number_of_neurons.append(num_neurons)
            # NOTE: store the sum across this batch (we will divide by total_samples later)
            spike_sum_over_samples.append(spikes_per_sample.sum().item())
        else:
            spike_sum_over_samples[idx] += spikes_per_sample.sum().item()


    # Hook every IF neuron
    for m in model.modules():
        if isinstance(m, IF):
            hooks.append(m.register_forward_hook(hook_fn))

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            current_batch_size = inputs.size(0)
            total_samples += current_batch_size

            # reset call counter each forward
            module_call_counter.clear()

            outputs = model(inputs)

            if outputs.dim() == 3:  # QCFS: [T, B, C]
                mean_out = outputs.mean(0)
            else:
                mean_out = outputs

            _, predicted = mean_out.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    for h in hooks:
        h.remove()

    # Normalize by total samples
    number_of_spikes = [s / total_samples for s in spike_sum_over_samples]

    # Build readable names
    inv_map = {v: k for k, v in layer_index.items()}
    layer_names = []
    for i in range(len(number_of_spikes)):
        mod_name, call_idx = inv_map[i]
        if call_idx > 1:
            layer_names.append(f"{mod_name}:{call_idx}")
        else:
            layer_names.append(mod_name)

    print("number_of_neurons:", number_of_neurons)
    print("number_of_spikes:", number_of_spikes)
    print("layer_names:", layer_names)

    final_acc = 100.0 * correct / total
    return final_acc, number_of_neurons, number_of_spikes, layer_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="cifar10")
    parser.add_argument("--arch", type=str, default="vgg16")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--outdir", type=str, default="spike_images",
                        help="directory to save spike plots")
    args = parser.parse_args()

    # === Load dataset ===
    train_loader, test_loader = datapool(args.data, batchsize=200)

    # === Build model ===
    model = modelpool(args.arch, args.data)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.set_T(args.T)
    model.set_L(args.L)

    # === Run test + spikes ===
    #acc, n_neurons, n_spikes = test_with_spikes(model, test_loader, device)
    acc, n_neurons, n_spikes, layer_names = test_with_spikes(model, test_loader, device)

    # === Ensure output directory exists ===
    os.makedirs(args.outdir, exist_ok=True)

    # === Auto image name ===
    img_name = f"{args.arch}_{args.data}_L{args.L}_T{args.T}_spikes.png"
    img_path = os.path.join(args.outdir, img_name)

    # === Plot histogram ===
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(n_spikes)), n_spikes, color="skyblue", edgecolor="black")
    plt.xticks(range(len(layer_names)), layer_names, rotation=90, fontsize=8)
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Average Spikes per Neuron", fontsize=12)
    plt.title(f"{args.arch.upper()} | {args.data} | L={args.L} | T={args.T} | Acc={acc:.2f}%", fontsize=14)
    plt.tight_layout()
    plt.savefig(img_path, dpi=300)
    plt.close()

    print(f"Saved spike histogram: {img_path}")
