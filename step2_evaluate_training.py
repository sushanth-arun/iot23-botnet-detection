#!/usr/bin/env python3
"""
Step 2: Model Training & Evaluation on 80/20 biased dataset
-----------------------------------------------------------
Trains LightGBM, XGBoost, and LSTM candidates on the 80/20 train log.
Outputs training scorecard metrics and saves candidates to local directory.
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve, auc

# Optional ML libraries
try:
    import xgboost as xgb
except ImportError:
    print("[!] XGBoost not installed. Run: pip install xgboost")
    sys.exit(1)

try:
    import lightgbm as lgb
except ImportError:
    print("[!] LightGBM not installed. Run: pip install lightgbm")
    sys.exit(1)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[!] PyTorch not found. LSTM training will be skipped.")

# Import wrapper class
from lstm_wrapper import LSTMClassifier, LSTMDeploymentWrapper

def main():
    parser = argparse.ArgumentParser(description="Train network models on 80/20 biased split.")
    parser.add_argument('--epochs', type=int, default=3, help="LSTM training epochs (default: 3).")
    args = parser.parse_args()
    
    train_path = "conn.log.train_80_20"
    if not os.path.exists(train_path):
        print(f"[!] Error: Training dataset '{train_path}' not found. Run generator first.")
        sys.exit(1)
        
    print("[+] Loading 80/20 biased training log...")
    df = pd.read_csv(train_path, sep='\t', low_memory=False).dropna(subset=['label'])
    
    # Feature columns
    numeric_cols = ['duration', 'orig_bytes', 'resp_bytes', 'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
    categorical_cols = ['proto', 'service', 'conn_state', 'history']
    
    X = df[numeric_cols + categorical_cols].copy()
    labels = df['label'].astype(str).str.strip().str.lower()
    y = (~labels.str.startswith('benign')).astype(int).values
    
    # Preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_cols),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='unknown')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_cols)
    ])
    
    print("[+] Fitting preprocessing pipeline on training data...")
    X_proc = preprocessor.fit_transform(X)
    joblib.dump(preprocessor, 'preprocessor.joblib')
    
    ratio = (len(y) - sum(y)) / sum(y) if sum(y) > 0 else 1.0
    print(f"    - Training shape: {X_proc.shape}")
    print(f"    - Class Balance (Benign/Malicious ratio): {ratio:.4f}")
    
    scorecard_data = []
    
    # --- 1. LightGBM ---
    print("\n[+] Training LightGBM Classifier...")
    clf_lgb = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, scale_pos_weight=ratio, random_state=42, n_jobs=-1, verbosity=-1)
    t0 = time.time()
    clf_lgb.fit(X_proc, y)
    lgb_time = time.time() - t0
    
    preds_lgb = clf_lgb.predict(X_proc)
    probs_lgb = clf_lgb.predict_proba(X_proc)[:, 1]
    
    f1_lgb = f1_score(y, preds_lgb, zero_division=0)
    acc_lgb = accuracy_score(y, preds_lgb)
    prec_lgb = precision_score(y, preds_lgb, zero_division=0)
    rec_lgb = recall_score(y, preds_lgb, zero_division=0)
    
    roc_auc_lgb = roc_auc_score(y, probs_lgb)
    prec_curve_lgb, rec_curve_lgb, _ = precision_recall_curve(y, probs_lgb)
    pr_auc_lgb = auc(rec_curve_lgb, prec_curve_lgb)
    
    # Save candidate
    pipeline_lgb = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf_lgb)])
    joblib.dump(pipeline_lgb, 'candidate_lgb.joblib')
    scorecard_data.append(("LightGBM", f1_lgb, acc_lgb, prec_lgb, rec_lgb, roc_auc_lgb, pr_auc_lgb, lgb_time))
    
    # --- 2. XGBoost ---
    print("[+] Training XGBoost Classifier...")
    clf_xgb = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, scale_pos_weight=ratio, random_state=42, n_jobs=-1, eval_metric='logloss')
    t0 = time.time()
    clf_xgb.fit(X_proc, y)
    xgb_time = time.time() - t0
    
    preds_xgb = clf_xgb.predict(X_proc)
    probs_xgb = clf_xgb.predict_proba(X_proc)[:, 1]
    
    f1_xgb = f1_score(y, preds_xgb, zero_division=0)
    acc_xgb = accuracy_score(y, preds_xgb)
    prec_xgb = precision_score(y, preds_xgb, zero_division=0)
    rec_xgb = recall_score(y, preds_xgb, zero_division=0)
    
    roc_auc_xgb = roc_auc_score(y, probs_xgb)
    prec_curve_xgb, rec_curve_xgb, _ = precision_recall_curve(y, probs_xgb)
    pr_auc_xgb = auc(rec_curve_xgb, prec_curve_xgb)
    
    # Save candidate
    pipeline_xgb = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf_xgb)])
    joblib.dump(pipeline_xgb, 'candidate_xgb.joblib')
    scorecard_data.append(("XGBoost", f1_xgb, acc_xgb, prec_xgb, rec_xgb, roc_auc_xgb, pr_auc_xgb, xgb_time))
    
    # --- 3. LSTM (PyTorch) ---
    if TORCH_AVAILABLE:
        print("[+] Preparing sequences and training LSTM Classifier...")
        # Helper to create sequences
        def create_sequences(X_data, y_data, seq_len=5):
            X_seq, y_seq = [], []
            for i in range(len(X_data) - seq_len + 1):
                X_seq.append(X_data[i : i + seq_len])
                y_seq.append(y_data[i + seq_len - 1])
            return np.array(X_seq), np.array(y_seq)
            
        X_seq, y_seq = create_sequences(X_proc, y, seq_len=5)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMClassifier(input_dim=X_proc.shape[1], hidden_dim=32)
        model.to(device)
        
        loader = DataLoader(TensorDataset(torch.FloatTensor(X_seq), torch.FloatTensor(y_seq)), batch_size=512, shuffle=True)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        t0 = time.time()
        for epoch in range(args.epochs):
            model.train()
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
        lstm_time = time.time() - t0
        
        model.eval()
        with torch.no_grad():
            probs = model(torch.FloatTensor(X_seq).to(device)).cpu().numpy()
        preds_lstm = (probs >= 0.90).astype(int)
        
        f1_lstm = f1_score(y_seq, preds_lstm, zero_division=0)
        acc_lstm = accuracy_score(y_seq, preds_lstm)
        prec_lstm = precision_score(y_seq, preds_lstm, zero_division=0)
        rec_lstm = recall_score(y_seq, preds_lstm, zero_division=0)
        
        roc_auc_lstm = roc_auc_score(y_seq, probs)
        prec_curve_lstm, rec_curve_lstm, _ = precision_recall_curve(y_seq, probs)
        pr_auc_lstm = auc(rec_curve_lstm, prec_curve_lstm)
        
        # Save candidate
        pipeline_lstm = LSTMDeploymentWrapper(
            state_dict=model.state_dict(),
            input_dim=X_proc.shape[1],
            preprocessor=preprocessor,
            seq_len=5,
            hidden_dim=32
        )
        joblib.dump(pipeline_lstm, 'candidate_lstm.joblib')
        scorecard_data.append(("LSTM", f1_lstm, acc_lstm, prec_lstm, rec_lstm, roc_auc_lstm, pr_auc_lstm, lstm_time))
        
        # Save confusion matrix for LSTM (training fitting)
        cm = confusion_matrix(y_seq, preds_lstm)
        plt.figure(figsize=(5, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('LSTM Training Confusion Matrix')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig('confusion_matrix_lstm_training.png')
        plt.close()

    # --- Print Scorecard Table ---
    print("\n" + "=" * 125)
    print(" TRAINING SPLIT PERFORMANCE SCORECARD (SECTION 7 COMPLIANCE) ".center(125, "="))
    print("=" * 125)
    print(f"| {'Classifier':<15} | {'F1-Score':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'ROC-AUC':<10} | {'PR-AUC':<10} | {'Train Time':<12} |")
    print("-" * 125)
    for row in scorecard_data:
        print(f"| {row[0]:<15} | {row[1]:<10.4f} | {row[2]:<10.4f} | {row[3]:<10.4f} | {row[4]:<10.4f} | {row[5]:<10.4f} | {row[6]:<10.4f} | {row[7]:<10.2f}s |")
    print("=" * 125)
    print("[+] Step 2 finished successfully.")

if __name__ == '__main__':
    main()
