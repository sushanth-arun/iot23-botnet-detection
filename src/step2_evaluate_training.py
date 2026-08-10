#!/usr/bin/env python3
# Step 2: Train LightGBM, XGBoost, and PyTorch LSTM models on the training dataset.

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

# Import optional ML libraries
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
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[!] PyTorch not found. Neural models will be skipped.")

# PyTorch GRU Model Architecture
class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out).squeeze(-1)

# PyTorch 1D Temporal Convolutional Network (TCN) Architecture
class TCNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(TCNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=2, padding=1, dilation=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2, padding=2, dilation=2)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = out[:, :, -1]
        out = self.fc(out)
        return self.sigmoid(out)

# Import LSTM wrapper
from lstm_wrapper import LSTMClassifier, LSTMDeploymentWrapper

def find_path(folder, filename):
    candidates = [
        os.path.join(folder, filename),
        os.path.join("..", folder, filename),
        filename,
        os.path.join("..", filename)
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join(folder, filename)

def main():
    parser = argparse.ArgumentParser(description="Train network models on 80/20 biased split.")
    parser.add_argument('--epochs', type=int, default=3, help="LSTM training epochs (default: 3).")
    args = parser.parse_args()
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    train_path = find_path("datasets", "conn.log.train_20_80")
    if not os.path.exists(train_path):
        print(f"[!] Error: Training dataset '{train_path}' not found. Run generator first.")
        sys.exit(1)
        
    print("[+] Loading 20/80 biased training log...")
    df = pd.read_csv(train_path, sep='\t', low_memory=False).dropna(subset=['label'])
    
    # Define feature columns
    numeric_cols = ['duration', 'orig_bytes', 'resp_bytes', 'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
    categorical_cols = ['proto', 'service', 'conn_state', 'history']
    
    X = df[numeric_cols + categorical_cols].copy()
    labels = df['label'].astype(str).str.strip().str.lower()
    y = (~labels.str.startswith('benign')).astype(int).values
    
    # Define preprocessing pipeline
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_cols),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='unknown')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_cols)
    ])
    
    print("[+] Fitting preprocessing pipeline on training data...")
    X_proc = preprocessor.fit_transform(X)
    joblib.dump(preprocessor, os.path.join('models', 'preprocessor.joblib'))
    
    ratio = (len(y) - sum(y)) / sum(y) if sum(y) > 0 else 1.0
    print(f"    - Training shape: {X_proc.shape}")
    print(f"    - Class Balance (Benign/Malicious ratio): {ratio:.4f}")
    
    scorecard_data = []
    
    # Train LightGBM
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
    
    # Save LightGBM candidate
    pipeline_lgb = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf_lgb)])
    joblib.dump(pipeline_lgb, os.path.join('models', 'candidate_lgb.joblib'))
    scorecard_data.append(("LightGBM", f1_lgb, acc_lgb, prec_lgb, rec_lgb, roc_auc_lgb, pr_auc_lgb, lgb_time))
    
    # Train XGBoost
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
    
    # Save XGBoost candidate
    pipeline_xgb = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf_xgb)])
    joblib.dump(pipeline_xgb, os.path.join('models', 'candidate_xgb.joblib'))
    scorecard_data.append(("XGBoost", f1_xgb, acc_xgb, prec_xgb, rec_xgb, roc_auc_xgb, pr_auc_xgb, xgb_time))
    
    # Train CatBoost
    if CATBOOST_AVAILABLE:
        print("[+] Training CatBoost Classifier...")
        clf_cat = CatBoostClassifier(iterations=50, learning_rate=0.05, depth=6, verbose=0, random_seed=42)
        t0 = time.time()
        clf_cat.fit(X_proc, y)
        cat_time = time.time() - t0
        
        preds_cat = clf_cat.predict(X_proc)
        probs_cat = clf_cat.predict_proba(X_proc)[:, 1]
        
        f1_cat = f1_score(y, preds_cat, zero_division=0)
        acc_cat = accuracy_score(y, preds_cat)
        prec_cat = precision_score(y, preds_cat, zero_division=0)
        rec_cat = recall_score(y, preds_cat, zero_division=0)
        
        roc_auc_cat = roc_auc_score(y, probs_cat)
        prec_curve_cat, rec_curve_cat, _ = precision_recall_curve(y, probs_cat)
        pr_auc_cat = auc(rec_curve_cat, prec_curve_cat)
        
        pipeline_cat = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf_cat)])
        joblib.dump(pipeline_cat, os.path.join('models', 'candidate_cat.joblib'))
        scorecard_data.append(("CatBoost", f1_cat, acc_cat, prec_cat, rec_cat, roc_auc_cat, pr_auc_cat, cat_time))

    # Train PyTorch Sequence Models (LSTM, GRU, TCN)
    if TORCH_AVAILABLE:
        # Create sequential sliding windows
        def create_sequences(X_data, y_data, seq_len=5):
            X_seq, y_seq = [], []
            for i in range(len(X_data) - seq_len + 1):
                X_seq.append(X_data[i : i + seq_len])
                y_seq.append(y_data[i + seq_len - 1])
            return np.array(X_seq), np.array(y_seq)
            
        X_seq, y_seq = create_sequences(X_proc, y, seq_len=5)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Train PyTorch LSTM
        print("[+] Preparing sequences and training LSTM Classifier...")
        model_lstm = LSTMClassifier(input_dim=X_proc.shape[1], hidden_dim=32).to(device)
        loader = DataLoader(TensorDataset(torch.FloatTensor(X_seq), torch.FloatTensor(y_seq)), batch_size=512, shuffle=True)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model_lstm.parameters(), lr=0.001)
        
        t0 = time.time()
        for epoch in range(args.epochs):
            model_lstm.train()
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(model_lstm(bx), by)
                loss.backward()
                optimizer.step()
        lstm_time = time.time() - t0
        
        model_lstm.eval()
        with torch.no_grad():
            probs_lstm = model_lstm(torch.FloatTensor(X_seq).to(device)).cpu().numpy()
        preds_lstm = (probs_lstm >= 0.50).astype(int)
        
        f1_lstm = f1_score(y_seq, preds_lstm, zero_division=0)
        acc_lstm = accuracy_score(y_seq, preds_lstm)
        prec_lstm = precision_score(y_seq, preds_lstm, zero_division=0)
        rec_lstm = recall_score(y_seq, preds_lstm, zero_division=0)
        roc_auc_lstm = roc_auc_score(y_seq, probs_lstm)
        prec_c_l, rec_c_l, _ = precision_recall_curve(y_seq, probs_lstm)
        pr_auc_lstm = auc(rec_c_l, prec_c_l)
        
        pipeline_lstm = LSTMDeploymentWrapper(
            state_dict=model_lstm.state_dict(),
            input_dim=X_proc.shape[1],
            preprocessor=preprocessor,
            seq_len=5,
            hidden_dim=32
        )
        joblib.dump(pipeline_lstm, os.path.join('models', 'candidate_lstm.joblib'))
        scorecard_data.append(("LSTM", f1_lstm, acc_lstm, prec_lstm, rec_lstm, roc_auc_lstm, pr_auc_lstm, lstm_time))

        # 2. Train PyTorch GRU
        print("[+] Preparing sequences and training GRU Classifier...")
        model_gru = GRUClassifier(input_dim=X_proc.shape[1], hidden_dim=32).to(device)
        optimizer_gru = torch.optim.Adam(model_gru.parameters(), lr=0.001)
        t0 = time.time()
        for epoch in range(args.epochs):
            model_gru.train()
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer_gru.zero_grad()
                loss = criterion(model_gru(bx), by)
                loss.backward()
                optimizer_gru.step()
        gru_time = time.time() - t0
        
        model_gru.eval()
        with torch.no_grad():
            probs_gru = model_gru(torch.FloatTensor(X_seq).to(device)).cpu().numpy()
        preds_gru = (probs_gru >= 0.50).astype(int)
        
        f1_gru = f1_score(y_seq, preds_gru, zero_division=0)
        acc_gru = accuracy_score(y_seq, preds_gru)
        prec_gru = precision_score(y_seq, preds_gru, zero_division=0)
        rec_gru = recall_score(y_seq, preds_gru, zero_division=0)
        roc_auc_gru = roc_auc_score(y_seq, probs_gru)
        prec_c_g, rec_c_g, _ = precision_recall_curve(y_seq, probs_gru)
        pr_auc_gru = auc(rec_c_g, prec_c_g)
        
        pipeline_gru = LSTMDeploymentWrapper(
            state_dict=model_gru.state_dict(),
            input_dim=X_proc.shape[1],
            preprocessor=preprocessor,
            seq_len=5,
            hidden_dim=32,
            model_type='gru'
        )
        joblib.dump(pipeline_gru, os.path.join('models', 'candidate_gru.joblib'))
        scorecard_data.append(("GRU", f1_gru, acc_gru, prec_gru, rec_gru, roc_auc_gru, pr_auc_gru, gru_time))
        
        # Save training confusion matrix
        cm = confusion_matrix(y_seq, preds_lstm)
        plt.figure(figsize=(5, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('LSTM Training confusion matrix')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join('reports', 'confusion_matrix_lstm_training.png'))
        plt.close()
        plt.close()

    # Print performance metrics
    print("\n--- TRAINING SPLIT PERFORMANCE SCORECARD ---")
    print(f"{'Classifier':<15} {'F1-Score':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'ROC-AUC':<10} {'PR-AUC':<10} {'Train Time':<12}")
    for row in scorecard_data:
        print(f"{row[0]:<15} {row[1]:<10.4f} {row[2]:<10.4f} {row[3]:<10.4f} {row[4]:<10.4f} {row[5]:<10.4f} {row[6]:<10.4f} {row[7]:<10.2f}s")
    print("[+] Step 2 finished successfully.")

if __name__ == '__main__':
    main()

