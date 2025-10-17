#!/bin/bash
# Bash script to run QCFS spike rate experiments and save per-layer plots

# Dataset
DATASET="cifar10"
GPU=0
L=8   # quantization steps (must match training)

# Models to evaluate
MODELS=("vgg16" "resnet18" "resnet20")

# Time steps to test
TIMESTEPS=(2 4 8 16 32 64)

# Output directory for plots
OUTDIR="spike_images"
mkdir -p $OUTDIR

for MODEL in "${MODELS[@]}"; do
    echo "======================================="
    echo "Evaluating $MODEL on $DATASET with L=$L"
    echo "======================================="

    # Identifier for checkpoints (default suffix = n)
    ID="${MODEL}_L[${L}]_n"
    CKPT_PATH="${DATASET}-checkpoints/${ID}.pth"

    echo "Using checkpoint: $CKPT_PATH"

    for T in "${TIMESTEPS[@]}"; do
        echo "Running spike rate analysis for $MODEL with T=$T"
        CUDA_VISIBLE_DEVICES=$GPU python qcfs_spike_rate.py \
            --data $DATASET \
            --arch $MODEL \
            --checkpoint $CKPT_PATH \
            --T $T \
            --L $L \
            --outdir $OUTDIR
    done
done

echo "All spike rate plots saved in $OUTDIR/"
