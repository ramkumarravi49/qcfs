#!/bin/bash
# Bash script to run QCFS experiments and log results to CSV
# Sweeps over quantization levels and timesteps, logs both accuracy and avg_spikes

# Dataset
DATASET="cifar10"
GPU=1

# Quantization steps to train with
QUANT_LEVELS=(2 4 8 16 32)

# Models to train
MODELS=("vgg16" "resnet18" "resnet20")

# Time steps to test
TIMESTEPS=(2 4 8 16 32 64)

# Output CSV file
CSV_FILE="qcfs_results_all.csv"

# Write CSV header
echo "model,L,T,accuracy,avg_spikes" > $CSV_FILE

for L in "${QUANT_LEVELS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
      echo "======================================="
      echo "Training $MODEL on $DATASET with L=$L"
      echo "======================================="

      # Train ANN version (T=0)
      python main_train.py -data $DATASET -arch $MODEL -L $L -dev $GPU -T 0

      # Identifier used for checkpoints (default suffix = n)
      ID="${MODEL}_L[${L}]_n"

      echo "Checkpoint identifier: $ID"

      # Evaluate at each timestep
      for T in "${TIMESTEPS[@]}"; do
          echo "Testing $MODEL with L=$L and T=$T"
          # Capture output from main_test.py
          OUTPUT=$(python main_test.py -data $DATASET -arch $MODEL -id $ID -T $T -dev $GPU)

          # Expect output like: (accuracy, avg_spikes)
          ACC=$(echo $OUTPUT | sed 's/[(),]//g' | awk '{print $1}')
          SPIKES=$(echo $OUTPUT | sed 's/[(),]//g' | awk '{print $2}')

          echo "$MODEL,$L,$T,$ACC,$SPIKES" >> $CSV_FILE
      done
  done
done

echo "All results saved in $CSV_FILE"
