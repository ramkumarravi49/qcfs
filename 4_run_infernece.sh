#!/bin/bash
# Script to evaluate all trained QCFS checkpoints and log results

DATASET="cifar10"
GPU=1  # change if needed
TIMESTEPS=(2 4 8 16 32 64)
CSV_FILE="qcfs_inference_results.csv"

# Write CSV header
echo "model,L,T,accuracy,avg_spikes" > $CSV_FILE

# Loop through all available checkpoints
for CKPT in cifar10-checkpoints/*.pth; do
    # Extract filename, e.g. vgg16_L[8]_n.pth
    FNAME=$(basename $CKPT)
    
    # Parse model and L
    MODEL=$(echo $FNAME | cut -d'_' -f1)             # e.g. vgg16
    L=$(echo $FNAME | grep -oP '(?<=L\[)[0-9]+')     # e.g. 8
    ID="${FNAME%.pth}"                               # identifier without .pth

    echo "======================================="
    echo "Evaluating $MODEL (L=$L) checkpoint $FNAME"
    echo "======================================="

    for T in "${TIMESTEPS[@]}"; do
        echo "Testing $MODEL with L=$L and T=$T"

        # Print the Python command being executed
        echo "python main_test.py -data $DATASET -arch $MODEL -id $ID -L $L -T $T -dev $GPU"

        # Run the Python command
        OUTPUT=$(python main_test.py -data $DATASET -arch $MODEL -id $ID -L $L -T $T -dev $GPU)

        # Ensure OUTPUT prints "acc avg_spikes"
        ACC=$(echo $OUTPUT | awk '{print $1}')
        SPIKES=$(echo $OUTPUT | awk '{print $2}')

        echo "$MODEL,$L,$T,$ACC,$SPIKES" >> $CSV_FILE
    done
done

echo "All inference results saved in $CSV_FILE"
