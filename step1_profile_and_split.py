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

def main():
    local_train = "conn.log.train_20_80"
    local_test = "conn.log.test_90_10"
    local_cal = "conn.log.calibration_60_40"
    
    # Profile existing local files or sample from raw splits
    if os.path.exists(local_train) and os.path.exists(local_test) and (os.path.exists(local_cal) or os.path.exists("conn.log.calibration_90_10")):
        print("[+] Found pre-generated Zeek log splits in local directory. Profiling directly...")
        train_df = pd.read_csv(local_train, sep='\t', low_memory=False).dropna(subset=['label'])
        test_df = pd.read_csv(local_test, sep='\t', low_memory=False).dropna(subset=['label'])
        
        if not os.path.exists(local_cal) and os.path.exists("conn.log.calibration_90_10"):
            print("[+] Slicing conn.log.calibration_90_10 into conn.log.calibration_60_40 (60/40 split)...")
            temp_df = pd.read_csv("conn.log.calibration_90_10", sep='\t', low_memory=False).dropna(subset=['label'])
            temp_df['is_benign'] = temp_df['label'].astype(str).str.strip().str.lower().str.startswith('benign')
            ben_df = temp_df[temp_df['is_benign']]
            att_df = temp_df[~temp_df['is_benign']]
            # Sample 60:40 ratio for calibration split
            ben_sampled = ben_df.sample(n=1500, random_state=42)
            att_sampled = att_df.sample(n=1000, random_state=42)
            val_df = pd.concat([ben_sampled, att_sampled]).sort_index()
            val_df.drop(columns=['is_benign'], errors='ignore').to_csv(local_cal, sep="\t", index=False)
        else:
            val_df = pd.read_csv(local_cal, sep='\t', low_memory=False).dropna(subset=['label'])
            
        train_df['is_benign'] = train_df['label'].astype(str).str.strip().str.lower().str.startswith('benign')
        test_df['is_benign'] = test_df['label'].astype(str).str.strip().str.lower().str.startswith('benign')
        val_df['is_benign'] = val_df['label'].astype(str).str.strip().str.lower().str.startswith('benign')
    else:
        # Load raw dataset splits
        parent_processed_dir = "processed_data"
        train_src = os.path.join(parent_processed_dir, "train.csv")
        test_src = os.path.join(parent_processed_dir, "test.csv")
        val_src = os.path.join(parent_processed_dir, "val.csv")
        
        if not os.path.exists(train_src) or not os.path.exists(test_src) or not os.path.exists(val_src):
            parent_processed_dir = "../processed_data"
            train_src = os.path.join(parent_processed_dir, "train.csv")
            test_src = os.path.join(parent_processed_dir, "test.csv")
            val_src = os.path.join(parent_processed_dir, "val.csv")
            
        if not os.path.exists(train_src) or not os.path.exists(test_src) or not os.path.exists(val_src):
            print("[!] Error: Source splits (train.csv, test.csv, val.csv) not found in 'processed_data/' or '../processed_data/'.")
            print("    Please make sure the files are placed correctly.")
            sys.exit(1)
            
        print("[+] Loading parent split files for profiling and sampling from:", parent_processed_dir)
        train_df = pd.read_csv(train_src, low_memory=False).dropna(subset=['label'])
        test_df = pd.read_csv(test_src, low_memory=False).dropna(subset=['label'])
        val_df = pd.read_csv(val_src, low_memory=False).dropna(subset=['label'])
        
        # Convert labels to boolean flags
        train_df['is_benign'] = train_df['label'].astype(str).str.strip().str.lower().str.startswith('benign')
        test_df['is_benign'] = test_df['label'].astype(str).str.strip().str.lower().str.startswith('benign')
        val_df['is_benign'] = val_df['label'].astype(str).str.strip().str.lower().str.startswith('benign')
        
        print("\n[+] Generating custom Zeek-formatted log splits in local folder...")
        # Save sampled splits
        train_df.drop(columns=['is_benign'], errors='ignore').to_csv(local_train, sep="\t", index=False)
        
        te_ben = test_df[test_df['is_benign']]
        te_att = test_df[~test_df['is_benign']]
        te_ben_sampled = te_ben.sample(n=10000, random_state=42)
        te_att_sampled = te_att.sample(n=1111, random_state=42)
        pd.concat([te_ben_sampled, te_att_sampled]).sort_index().drop(columns=['is_benign'], errors='ignore').to_csv(local_test, sep="\t", index=False)
        
        val_ben = val_df[val_df['is_benign']]
        val_att = val_df[~val_df['is_benign']]
        cal_ben_sampled = val_ben.sample(n=6000, random_state=42)
        cal_att_sampled = val_att.sample(n=4000, random_state=42)
        pd.concat([cal_ben_sampled, cal_att_sampled]).sort_index().drop(columns=['is_benign'], errors='ignore').to_csv(local_cal, sep="\t", index=False)

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
