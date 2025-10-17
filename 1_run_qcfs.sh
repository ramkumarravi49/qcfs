#!/bin/bash
# Bash script to run QCFS experiments and log results to CSV

# Dataset
DATASET="cifar10"
GPU=1
L=8   # quantization step

# Models to train
MODELS=("vgg16" "resnet18" "resnet20")

# Time steps to test
TIMESTEPS=(2 4 8 16 32 64)

# Output CSV file
CSV_FILE="qcfs_results.csv"

# Write CSV header
echo "model,L,T,accuracy" > $CSV_FILE

for MODEL in "${MODELS[@]}"; do
    echo "======================================="
    echo "Training $MODEL on $DATASET with L=$L"
    echo "======================================="

    # Train ANN version (T=0)
    python main_train.py -data $DATASET -arch $MODEL -L $L -dev $GPU -T 0

    # Identifier used for checkpoints (default suffix = n)
    ID="${MODEL}_L[${L}]_n"

    echo "Checkpoint identifier: $ID"

    # Now evaluate at each timestep
    for T in "${TIMESTEPS[@]}"; do
        echo "Testing $MODEL with T=$T"
        # Capture accuracy from Python script
        ACC=$(python main_test.py -data $DATASET -arch $MODEL -id $ID -T $T -dev $GPU | tail -n 1)
        echo "$MODEL,$L,$T,$ACC" >> $CSV_FILE
    done
done

echo "All results saved in $CSV_FILE"
