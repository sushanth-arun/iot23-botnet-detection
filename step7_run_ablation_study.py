#!/usr/bin/env python3
"""
Step 7: Dynamic Feature & Capacity Ablation Testing
---------------------------------------------------
Performs structural ablation experiments (features removal, state flagging,
model capacity restrictions, and chronological data leakage) on the production
winner model loaded from model.joblib. Displays a compact consolidated metrics 
scorecard and saves a comparative visualization graph.
"""

import os
import sys
import time
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score, confusion_matrix

# Optional ML libraries
try:
    import xgboost as xgb
except ImportError:
    pass

try:
    import lightgbm as lgb
except ImportError:
    pass

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import LSTM wrapper classes
from lstm_wrapper import LSTMClassifier, LSTMDeploymentWrapper

# Helper to create sequences
def create_sequences(X, y, seq_len=5):
    X_seq = []
    y_seq = []
    for i in range(len(X) - seq_len + 1):
        X_seq.append(X[i : i + seq_len])
        y_seq.append(y[i + seq_len - 1])
    return np.array(X_seq), np.array(y_seq)

def train_pytorch_model(model, X_train, y_train, epochs=3, batch_size=256):
    if not TORCH_AVAILABLE:
        return None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=False)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    return model

def run_predictions(model, X_test, model_type, is_lstm=False):
    t0 = time.time()
    if is_lstm:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(device)
            probs = model(X_tensor).cpu().numpy()
        y_pred = (probs >= 0.90).astype(int)
    else:
        y_pred = model.predict(X_test)
    latency_ms = ((time.time() - t0) / len(X_test)) * 1000
    return y_pred, latency_ms

def main():
    test_path = "conn.log.test_80_20"
    train_path = "conn.log.train_80_20"
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Error: Datasets conn.log.train_80_20 and conn.log.test_80_20 not found.")
        sys.exit(1)
        
    if not os.path.exists('model.joblib'):
        print("[!] Error: Production model.joblib not found. Run Step 2 and Step 3 first.")
        sys.exit(1)
        
    # --- Load Production Model ---
    pipeline = joblib.load('model.joblib')
    
    # Detect Winner Model Type
    model_type = 'xgboost'
    is_lstm = False
    if hasattr(pipeline, 'state_dict'):
        model_type = 'lstm'
        is_lstm = True
        pipeline._lazy_init_model()
        baseline_model = pipeline._model
        preprocessor = pipeline.preprocessor
    else:
        baseline_model = pipeline.named_steps['classifier']
        preprocessor = pipeline.named_steps['preprocessor']
        clf_name = type(baseline_model).__name__
        if 'LGBM' in clf_name:
            model_type = 'lightgbm'
        elif 'XGB' in clf_name:
            model_type = 'xgboost'
            
    print(f"[+] Loaded production model. joblib detects active type: {model_type.upper()}")
    print(f"[+] Running comparative ablation study on {model_type.upper()}...")
    
    train_df = pd.read_csv(train_path, sep='\t', low_memory=False).dropna(subset=['label'])
    test_df = pd.read_csv(test_path, sep='\t', low_memory=False).dropna(subset=['label'])
    
    # Feature Sets
    numeric_cols = ['duration', 'orig_bytes', 'resp_bytes', 'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
    categorical_cols = ['proto', 'service', 'conn_state', 'history']
    
    def extract_features_labels(df):
        X = df[numeric_cols + categorical_cols].copy()
        labels = df['label'].astype(str).str.strip().str.lower()
        y = (~labels.str.startswith('benign')).astype(int).values
        return X, y
        
    X_train, y_train = extract_features_labels(train_df)
    X_test, y_test = extract_features_labels(test_df)
    
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    if is_lstm:
        X_test_eval, y_test_eval = create_sequences(X_test_proc, y_test, seq_len=5)
    else:
        X_test_eval = X_test_proc
        y_test_eval = y_test
        
    # Evaluate Baseline using the production model weights
    print("\n[+] Baseline Model: Evaluates performance using all features with chronological temporal split.")
    y_pred_base, base_lat = run_predictions(baseline_model, X_test_eval, model_type, is_lstm)
    base_f1 = f1_score(y_test_eval, y_pred_base, zero_division=0)
    cm_base = confusion_matrix(y_test_eval, y_pred_base, labels=[0, 1])
    tn, fp, fn, tp = cm_base.ravel() if cm_base.size == 4 else (len(y_test_eval) - sum(y_test_eval), 0, 0, sum(y_test_eval))
    base_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # --- Experiment 1: No Volumetric (Zero out all numeric feature columns) ---
    print("[+] Experiment 1 (No Volumetric): Zeroes out all numeric features at test time.")
    print("    * Attributes removed: duration, orig_bytes, resp_bytes, orig_pkts, orig_ip_bytes, resp_pkts, resp_ip_bytes")
    X_test_exp1 = X_test_proc.copy()
    X_test_exp1[:, :8] = 0.0
    
    if is_lstm:
        X_test_eval_exp1, _ = create_sequences(X_test_exp1, y_test, seq_len=5)
    else:
        X_test_eval_exp1 = X_test_exp1
        
    y_pred_exp1, exp1_lat = run_predictions(baseline_model, X_test_eval_exp1, model_type, is_lstm)
    exp1_f1 = f1_score(y_test_eval, y_pred_exp1, zero_division=0)
    cm_exp1 = confusion_matrix(y_test_eval, y_pred_exp1, labels=[0, 1])
    tn, fp, fn, tp = cm_exp1.ravel() if cm_exp1.size == 4 else (len(y_test_eval), 0, 0, 0)
    exp1_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # --- Experiment 2: No Connection State (Zero out all categorical protocol flags) ---
    print("[+] Experiment 2 (No Conn State): Zeroes out all categorical features at test time.")
    print("    * Attributes removed: proto, service, conn_state, history")
    X_test_exp2 = X_test_proc.copy()
    X_test_exp2[:, 8:] = 0.0
    
    if is_lstm:
        X_test_eval_exp2, _ = create_sequences(X_test_exp2, y_test, seq_len=5)
    else:
        X_test_eval_exp2 = X_test_exp2
        
    y_pred_exp2, exp2_lat = run_predictions(baseline_model, X_test_eval_exp2, model_type, is_lstm)
    exp2_f1 = f1_score(y_test_eval, y_pred_exp2, zero_division=0)
    cm_exp2 = confusion_matrix(y_test_eval, y_pred_exp2, labels=[0, 1])
    tn, fp, fn, tp = cm_exp2.ravel() if cm_exp2.size == 4 else (len(y_test_eval), 0, 0, 0)
    exp2_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # --- Experiment 3: Hyperparameter/Capacity Restriction (Stump / Single Unit) ---
    print("[+] Experiment 3 (Capacity Limit): Restricts model capacity to a Decision Stump or single recurrent cell to evaluate underfitting.")
    exp3_f1, exp3_fpr, exp3_lat = 0.0, 0.0, 0.0
    ratio = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1.0
    
    if model_type == 'xgboost':
        stump = xgb.XGBClassifier(n_estimators=1, max_depth=1, scale_pos_weight=ratio, random_state=42, n_jobs=-1, eval_metric='logloss')
        stump.fit(X_train_proc, y_train)
        y_pred_exp3, exp3_lat = run_predictions(stump, X_test_eval, model_type, is_lstm)
        exp3_f1 = f1_score(y_test_eval, y_pred_exp3, zero_division=0)
        cm_exp3 = confusion_matrix(y_test_eval, y_pred_exp3, labels=[0, 1])
        tn, fp, fn, tp = cm_exp3.ravel() if cm_exp3.size == 4 else (len(y_test_eval), 0, 0, 0)
        exp3_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    elif model_type == 'lightgbm':
        stump = lgb.LGBMClassifier(n_estimators=1, max_depth=1, scale_pos_weight=ratio, random_state=42, n_jobs=-1, verbosity=-1)
        stump.fit(X_train_proc, y_train)
        y_pred_exp3, exp3_lat = run_predictions(stump, X_test_eval, model_type, is_lstm)
        exp3_f1 = f1_score(y_test_eval, y_pred_exp3, zero_division=0)
        cm_exp3 = confusion_matrix(y_test_eval, y_pred_exp3, labels=[0, 1])
        tn, fp, fn, tp = cm_exp3.ravel() if cm_exp3.size == 4 else (len(y_test_eval), 0, 0, 0)
        exp3_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    elif model_type == 'lstm':
        X_train_seq, y_train_seq = create_sequences(X_train_proc, y_train, seq_len=5)
        stump = LSTMClassifier(input_dim=X_train_proc.shape[1], hidden_dim=1)
        stump = train_pytorch_model(stump, X_train_seq, y_train_seq, epochs=3)
        y_pred_exp3, exp3_lat = run_predictions(stump, X_test_eval, model_type, is_lstm)
        exp3_f1 = f1_score(y_test_eval, y_pred_exp3, zero_division=0)
        cm_exp3 = confusion_matrix(y_test_eval, y_pred_exp3, labels=[0, 1])
        tn, fp, fn, tp = cm_exp3.ravel() if cm_exp3.size == 4 else (len(y_test_eval), 0, 0, 0)
        exp3_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # --- Experiment 4: Temporal Split Violation (Shuffled Split Leakage) ---
    print("[+] Experiment 4 (Shuffled Split): Violates chronological splitting by training/testing on randomly shuffled data to show leakage.")
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    shuffled_df = combined_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_samples = len(shuffled_df)
    train_split_idx = int(0.70 * n_samples)
    
    shuff_train_df = shuffled_df.iloc[:train_split_idx]
    shuff_test_df = shuffled_df.iloc[train_split_idx:]
    
    X_shuff_tr, y_shuff_tr = extract_features_labels(shuff_train_df)
    X_shuff_te, y_shuff_te = extract_features_labels(shuff_test_df)
    
    preprocessor_shuff = Pipeline(steps=[
        ('preprocess', ColumnTransformer(transformers=[
            ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_cols),
            ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='unknown')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_cols)
        ], remainder='drop'))
    ])
    
    X_shuff_tr_proc = preprocessor_shuff.fit_transform(X_shuff_tr)
    X_shuff_te_proc = preprocessor_shuff.transform(X_shuff_te)
    
    ratio_shuff = (len(y_shuff_tr) - sum(y_shuff_tr)) / sum(y_shuff_tr) if sum(y_shuff_tr) > 0 else 1.0
    
    exp4_model = None
    if model_type == 'xgboost':
        exp4_model = xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=ratio_shuff, random_state=42, n_jobs=-1, eval_metric='logloss'
        )
        exp4_model.fit(X_shuff_tr_proc, y_shuff_tr)
        X_shuff_te_eval = X_shuff_te_proc
        y_shuff_te_eval = y_shuff_te
    elif model_type == 'lightgbm':
        exp4_model = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=ratio_shuff, random_state=42, n_jobs=-1, verbosity=-1
        )
        exp4_model.fit(X_shuff_tr_proc, y_shuff_tr)
        X_shuff_te_eval = X_shuff_te_proc
        y_shuff_te_eval = y_shuff_te
    elif model_type == 'lstm':
        X_shuff_tr_seq, y_shuff_tr_seq = create_sequences(X_shuff_tr_proc, y_shuff_tr, seq_len=5)
        X_shuff_te_seq, y_shuff_te_seq = create_sequences(X_shuff_te_proc, y_shuff_te, seq_len=5)
        
        exp4_model = LSTMClassifier(input_dim=X_shuff_tr_proc.shape[1], hidden_dim=32)
        exp4_model = train_pytorch_model(exp4_model, X_shuff_tr_seq, y_shuff_tr_seq, epochs=3)
        X_shuff_te_eval = X_shuff_te_seq
        y_shuff_te_eval = y_shuff_te_seq
        
    y_pred_exp4, exp4_lat = run_predictions(exp4_model, X_shuff_te_eval, model_type, is_lstm)
    exp4_f1 = f1_score(y_shuff_te_eval, y_pred_exp4, zero_division=0)
    cm_exp4 = confusion_matrix(y_shuff_te_eval, y_pred_exp4, labels=[0, 1])
    tn, fp, fn, tp = cm_exp4.ravel() if cm_exp4.size == 4 else (len(y_shuff_te_eval), 0, 0, 0)
    exp4_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # --- Consolidated Ablation Summary Table ---
    print("\n" + "=" * 105)
    print(f" CONSOLIDATED ABLATION SUMMARY TABLE ".center(105, "="))
    print("=" * 105)
    print(f"| {'Experiment Name':<30} | {'F1-Score':<10} | {'FPR':<10} | {'Latency':<12} | {'F1 Delta':<10} | {'FPR Delta':<10} |")
    print("-" * 105)
    print(f"| {'Baseline Model (Temporal)':<30} | {base_f1:<10.4f} | {base_fpr:<10.4f} | {base_lat:<9.4f}ms | {'Reference':<10} | {'Reference':<10} |")
    print(f"| {'Ablation 1 (No Volumetric)':<30} | {exp1_f1:<10.4f} | {exp1_fpr:<10.4f} | {exp1_lat:<9.4f}ms | {exp1_f1-base_f1:<+10.4f} | {exp1_fpr-base_fpr:<+10.4f} |")
    print(f"| {'Ablation 2 (No Conn State)':<30} | {exp2_f1:<10.4f} | {exp2_fpr:<10.4f} | {exp2_lat:<9.4f}ms | {exp2_f1-base_f1:<+10.4f} | {exp2_fpr-base_fpr:<+10.4f} |")
    print(f"| {'Ablation 3 (Capacity Limit)':<30} | {exp3_f1:<10.4f} | {exp3_fpr:<10.4f} | {exp3_lat:<9.4f}ms | {exp3_f1-base_f1:<+10.4f} | {exp3_fpr-base_fpr:<+10.4f} |")
    print(f"| {'Ablation 4 (Shuffled Split)':<30} | {exp4_f1:<10.4f} | {exp4_fpr:<10.4f} | {exp4_lat:<9.4f}ms | {exp4_f1-base_f1:<+10.4f} | {exp4_fpr-base_fpr:<+10.4f} |")
    print("=" * 105)

    # --- Save comparative bar chart ---
    try:
        experiments = ['Baseline', 'No Volumetric', 'No State Flags', 'Stump / Cap Limit', 'Shuffled Split']
        f1_scores = [base_f1, exp1_f1, exp2_f1, exp3_f1, exp4_f1]
        fpr_rates = [base_fpr, exp1_fpr, exp2_fpr, exp3_fpr, exp4_fpr]
        
        x = np.arange(len(experiments))
        width = 0.35
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Ablation Experiment')
        ax1.set_ylabel('F1-Score', color=color)
        rects1 = ax1.bar(x - width/2, f1_scores, width, label='F1-Score', color=color, alpha=0.8)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 1.1)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('False Positive Rate (FPR)', color=color)
        rects2 = ax2.bar(x + width/2, fpr_rates, width, label='FPR', color=color, alpha=0.8)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 1.1)
        
        plt.title(f'Ablation Study: Metrics Comparison ({model_type.upper()})')
        ax1.set_xticks(x)
        ax1.set_xticklabels(experiments, rotation=15)
        fig.tight_layout()
        
        filepath = 'ablation_study_comparison.png'
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"[+] Saved ablation study comparative bar chart to: {filepath}")
    except Exception as e:
        print(f"[!] Warning: Could not save ablation chart ({e})")

if __name__ == '__main__':
    main()
