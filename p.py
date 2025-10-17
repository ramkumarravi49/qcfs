#!/usr/bin/env python3
# save as clean_qcfs_csv.py and run: python clean_qcfs_csv.py

import re
import pandas as pd

SRC = "qcfs_results.csv"
DST = "qcfs_results_clean.csv"

rows = []
with open(SRC, "r", encoding="utf-8") as f:
    lines = f.read().splitlines()

# find header (first non-empty line)
start = 0
for i, line in enumerate(lines):
    if line.strip():
        start = i
        break

# parse each data line
for line in lines[start+1:]:
    if not line.strip():
        continue
    parts = line.split(",")
    if len(parts) < 4:
        # skip malformed lines
        continue
    model = parts[0].strip()
    L = parts[1].strip()
    T = parts[2].strip()
    # everything after the 3rd comma belongs to the accuracy field (which may include commas)
    acc_raw = ",".join(parts[3:]).strip()

    # extract numbers (floats or ints) from acc_raw robustly
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", acc_raw)
    if len(nums) >= 2:
        acc = float(nums[0])
        spikes = float(nums[1])
    elif len(nums) == 1:
        acc = float(nums[0])
        spikes = None
    else:
        acc = None
        spikes = None

    # Normalize types for L and T if numeric
    try:
        L_val = int(L)
    except:
        L_val = L
    try:
        T_val = int(T)
    except:
        T_val = T

    rows.append({
        "model": model,
        "L": L_val,
        "T": T_val,
        "accuracy": acc,
        "avg_spikes": spikes
    })

# Build DataFrame and save
df = pd.DataFrame(rows, columns=["model","L","T","accuracy","avg_spikes"])
df.to_csv(DST, index=False)
print(f"Wrote cleaned CSV to: {DST}")
print(df.head(12).to_string(index=False))
