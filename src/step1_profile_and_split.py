#!/usr/bin/env python3
# Step 1: Profile dataset splits and create train, test, and calibration log files.

import os
import sys
import pandas as pd
import numpy as np

def calculate_imbalance_severity(benign, attack):
    total = benign + attack
    if total == 0:
        return "N/A", "N/A"
    majority = max(benign, attack)
    minority = min(benign, attack)
    ratio = majority / minority if minority > 0 else float('inf')
    
    if ratio > 10.0:
        severity = "Extreme Imbalance"
    elif ratio > 4.0:
        severity = "Severe Imbalance"
    elif ratio > 1.5:
        severity = "Moderate Imbalance"
    else:
        severity = "Low Imbalance"
        
    return f"1:{ratio:.2f}", severity

def find_dataset(filename):
    candidates = [
        os.path.join("datasets", filename),
        os.path.join("..", "datasets", filename),
        filename,
        os.path.join("..", filename)
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join("datasets", filename)

def main():
    os.makedirs("datasets", exist_ok=True)
    local_train = find_dataset("conn.log.train_20_80")
    local_test = find_dataset("conn.log.test_90_10")
    local_cal = find_dataset("conn.log.calibration_60_40")
    
    # Profile existing local files directly from datasets directory
    if not (os.path.exists(local_train) and os.path.exists(local_test) and os.path.exists(local_cal)):
        print(f"[!] Error: Dataset splits not found in 'datasets/' folder.")
        print(f"    Expected: {local_train}, {local_test}, {local_cal}")
        sys.exit(1)
        
    print("[+] Found pre-generated Zeek log splits. Profiling directly...")
    train_df = pd.read_csv(local_train, sep='\t', low_memory=False).dropna(subset=['label'])
    test_df = pd.read_csv(local_test, sep='\t', low_memory=False).dropna(subset=['label'])
    val_df = pd.read_csv(local_cal, sep='\t', low_memory=False).dropna(subset=['label'])
        
    train_df['is_benign'] = train_df['label'].astype(str).str.strip().str.lower().str.startswith('benign')
    test_df['is_benign'] = test_df['label'].astype(str).str.strip().str.lower().str.startswith('benign')
    val_df['is_benign'] = val_df['label'].astype(str).str.strip().str.lower().str.startswith('benign')

    train_ben = len(train_df[train_df['is_benign']])
    train_att = len(train_df[~train_df['is_benign']])
    
    val_ben = len(val_df[val_df['is_benign']])
    val_att = len(val_df[~val_df['is_benign']])
    
    test_ben = len(test_df[test_df['is_benign']])
    test_att = len(test_df[~test_df['is_benign']])
    
    # Calculate imbalance statistics
    tr_ratio, tr_sev = calculate_imbalance_severity(train_ben, train_att)
    val_ratio, val_sev = calculate_imbalance_severity(val_ben, val_att)
    te_ratio, te_sev = calculate_imbalance_severity(test_ben, test_att)
    
    print("\n--- CLASS DISTRIBUTION & IMBALANCE SEVERITY ANALYSIS REPORT ---")
    print(f"{'Split File':<15} {'Total Samples':<15} {'Benign Samples':<16} {'Attack Samples':<16} {'Class Ratio':<15} {'Severity':<20}")
    print(f"{local_train:<15} {len(train_df):<15} {train_ben:<16} {train_att:<16} {tr_ratio:<15} {tr_sev:<20}")
    print(f"{local_cal:<15} {len(val_df):<15} {val_ben:<16} {val_att:<16} {val_ratio:<15} {val_sev:<20}")
    print(f"{local_test:<15} {len(test_df):<15} {test_ben:<16} {test_att:<16} {te_ratio:<15} {te_sev:<20}")
    
    print("\n[+] Verification of files in local folder:")
    for fn in [local_train, local_test, local_cal]:
        size = os.path.getsize(fn) / (1024 * 1024)
        print(f"    - {fn:<30} | Size: {size:.2f} MB")
        
    print("\n[+] Step 1 finished successfully.")

if __name__ == '__main__':
    main()
