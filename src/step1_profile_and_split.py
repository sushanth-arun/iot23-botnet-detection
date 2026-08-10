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
    local_train = find_dataset("train_dataset.csv")
    local_test = find_dataset("unseen_test_dataset_a.csv")
    local_cal = find_dataset("unseen_calibration_dataset_b.csv")
    
    print("[+] Reading dataset pool to generate clean balanced 50:50 Train, Calibration, and Test splits...")
    dfs = []
    for path in [local_train, local_test, local_cal]:
        if os.path.exists(path):
            dfs.append(pd.read_csv(path, sep='\t', low_memory=False).dropna(subset=['label']))
            
    full_df = pd.concat(dfs).reset_index(drop=True)
    full_df['is_benign'] = full_df['label'].astype(str).str.strip().str.lower().str.startswith('benign')
    
    ben_df = full_df[full_df['is_benign']].sample(frac=1.0, random_state=42).reset_index(drop=True)
    att_df = full_df[~full_df['is_benign']].sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # 1. Balanced Training Split (50:50 -> 30,000 Benign, 30,000 Attack)
    tr_ben = ben_df.iloc[:30000]
    tr_att = att_df.iloc[:30000]
    train_df = pd.concat([tr_ben, tr_att]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    train_df.drop(columns=['is_benign']).to_csv(local_train, sep='\t', index=False)
    
    # 2. Balanced Calibration Split (50:50 -> 2,000 Benign, 2,000 Attack)
    cal_ben = ben_df.iloc[30000:32000]
    cal_att = att_df.iloc[30000:32000]
    val_df = pd.concat([cal_ben, cal_att]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_df.drop(columns=['is_benign']).to_csv(local_cal, sep='\t', index=False)
    
    # 3. Balanced Test Split (50:50 -> 10,000 Benign, 10,000 Attack)
    te_ben = ben_df.iloc[32000:42000]
    te_att = att_df.iloc[32000:42000]
    test_df = pd.concat([te_ben, te_att]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    test_df.drop(columns=['is_benign']).to_csv(local_test, sep='\t', index=False)
    
    train_ben, train_att = len(tr_ben), len(tr_att)
    val_ben, val_att = len(cal_ben), len(cal_att)
    test_ben, test_att = len(te_ben), len(te_att)
    
    # Calculate imbalance statistics
    tr_ratio, tr_sev = calculate_imbalance_severity(train_ben, train_att)
    val_ratio, val_sev = calculate_imbalance_severity(val_ben, val_att)
    te_ratio, te_sev = calculate_imbalance_severity(test_ben, test_att)
    
    print("\n--- CLASS DISTRIBUTION & IMBALANCE SEVERITY ANALYSIS REPORT ---")
    print(f"{'Split File':<35} {'Total Samples':<15} {'Benign Samples':<16} {'Attack Samples':<16} {'Class Ratio':<15} {'Severity':<20}")
    print(f"{local_train:<35} {len(train_df):<15} {train_ben:<16} {train_att:<16} {tr_ratio:<15} {tr_sev:<20}")
    print(f"{local_cal:<35} {len(val_df):<15} {val_ben:<16} {val_att:<16} {val_ratio:<15} {val_sev:<20}")
    print(f"{local_test:<35} {len(test_df):<15} {test_ben:<16} {test_att:<16} {te_ratio:<15} {te_sev:<20}")
    
    print("\n[+] Verification of files in local folder:")
    for fn in [local_train, local_test, local_cal]:
        size = os.path.getsize(fn) / (1024 * 1024)
        print(f"    - {fn:<35} | Size: {size:.2f} MB")
        
    print("\n[+] Step 1 finished successfully.")

if __name__ == '__main__':
    main()
