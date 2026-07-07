#!/usr/bin/env python3
"""
Step 3: Unseen Testing & OOD Generalization Evaluation
------------------------------------------------------
Evaluates trained candidate models on the 80/20 test split (Dataset A) 
and the 90/10 calibration split (Dataset B). Chooses the best performing 
model, packages it, and saves it to local directory as model.joblib.
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve, auc

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def evaluate_predictions(y_true, y_pred, probs):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (len(y_true)-sum(y_true), 0, 0, sum(y_true))
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    try:
        roc_auc = roc_auc_score(y_true, probs)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, probs)
        pr_auc = auc(recall_curve, precision_curve)
    except:
        roc_auc = 0.50
        pr_auc = 0.50
        
    return acc, prec, rec, f1, roc_auc, pr_auc, fpr, fnr

def main():
    test_path = "conn.log.test_90_10"
    cal_path = "conn.log.calibration_90_10"
    
    if not os.path.exists(test_path) or not os.path.exists(cal_path):
        print("[!] Error: Custom test/calibration splits not found. Run Step 1/2 first.")
        sys.exit(1)
        
    print("[+] Loading test and calibration dataset logs...")
    test_df = pd.read_csv(test_path, sep='\t', low_memory=False).dropna(subset=['label'])
    cal_df = pd.read_csv(cal_path, sep='\t', low_memory=False).dropna(subset=['label'])
    
    # Feature columns
    numeric_cols = ['duration', 'orig_bytes', 'resp_bytes', 'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
    categorical_cols = ['proto', 'service', 'conn_state', 'history']
    
    # Extract features/labels
    def get_xy(df):
        X = df[numeric_cols + categorical_cols].copy()
        labels = df['label'].astype(str).str.strip().str.lower()
        y = (~labels.str.startswith('benign')).astype(int).values
        return X, y
        
    X_test, y_test = get_xy(test_df)
    X_cal, y_cal = get_xy(cal_df)
    
    # Load candidate checkpoints
    candidates = {}
    for name in ['lgb', 'xgb', 'lstm']:
        path = f"candidate_{name}.joblib"
        if os.path.exists(path):
            candidates[name] = joblib.load(path)
            
    if not candidates:
        print("[!] Error: No candidate models found. Run Step 2 first.")
        sys.exit(1)
        
    scorecard_test = []
    scorecard_cal = []
    
    for name, pipeline in candidates.items():
        is_lstm = (name == 'lstm')
        
        # 1. Evaluate on Dataset A (80/20 Test Log)
        if is_lstm:
            pipeline._lazy_init_model()
            X_p = pipeline.preprocessor.transform(X_test)
            X_seq, y_eval = [], []
            for i in range(len(X_p) - 4):
                X_seq.append(X_p[i : i + 5])
                y_eval.append(y_test[i + 4])
            X_eval = np.array(X_seq)
            y_eval = np.array(y_eval)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pipeline._model.to(device).eval()
            with torch.no_grad():
                probs = pipeline._model(torch.FloatTensor(X_eval).to(device)).cpu().numpy()
            preds = (probs >= 0.90).astype(int)
        else:
            X_eval = X_test
            y_eval = y_test
            preds = pipeline.predict(X_eval)
            probs = pipeline.predict_proba(X_eval)[:, 1]
            
        acc, prec, rec, f1, roc_auc, pr_auc, fpr, fnr = evaluate_predictions(y_eval, preds, probs)
        scorecard_test.append((name.upper(), f1, acc, prec, rec, roc_auc, pr_auc, fpr, fnr))
        
        # Save confusion matrix plot for internal test
        plt.figure(figsize=(5, 5))
        plt.imshow(confusion_matrix(y_eval, preds, labels=[0, 1]), interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'{name.upper()} Dataset A confusion matrix')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{name}_dataset_a.png')
        plt.close()
        
        # 2. Evaluate on Dataset B (90/10 Calibration Log)
        if is_lstm:
            X_p_cal = pipeline.preprocessor.transform(X_cal)
            X_seq_cal, y_eval_cal = [], []
            for i in range(len(X_p_cal) - 4):
                X_seq_cal.append(X_p_cal[i : i + 5])
                y_eval_cal.append(y_cal[i + 4])
            X_eval_cal = np.array(X_seq_cal)
            y_eval_cal = np.array(y_eval_cal)
            
            pipeline._model.to(device).eval()
            with torch.no_grad():
                probs_cal = pipeline._model(torch.FloatTensor(X_eval_cal).to(device)).cpu().numpy()
            preds_cal = (probs_cal >= 0.90).astype(int)
        else:
            X_eval_cal = X_cal
            y_eval_cal = y_cal
            preds_cal = pipeline.predict(X_eval_cal)
            probs_cal = pipeline.predict_proba(X_eval_cal)[:, 1]
            
        acc_c, prec_c, rec_c, f1_c, roc_auc_c, pr_auc_c, fpr_c, fnr_c = evaluate_predictions(y_eval_cal, preds_cal, probs_cal)
        scorecard_cal.append((name.upper(), f1_c, acc_c, prec_c, rec_c, roc_auc_c, pr_auc_c, fpr_c, fnr_c))
        
        # Save confusion matrix plot for external validation
        plt.figure(figsize=(5, 5))
        plt.imshow(confusion_matrix(y_eval_cal, preds_cal, labels=[0, 1]), interpolation='nearest', cmap=plt.cm.Oranges)
        plt.title(f'{name.upper()} Dataset B confusion matrix')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{name}_dataset_b.png')
        plt.close()

    # --- Print scorecards ---
    print("\n--- SCORECARD A: UNSEEN TEMPORAL TESTING (DATASET A) ---")
    print(f"{'Classifier':<15} {'F1-Score':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'ROC-AUC':<10} {'PR-AUC':<10} {'FPR':<10} {'FNR':<10}")
    for row in scorecard_test:
        print(f"{row[0]:<15} {row[1]:<10.4f} {row[2]:<10.4f} {row[3]:<10.4f} {row[4]:<10.4f} {row[5]:<10.4f} {row[6]:<10.4f} {row[7]:<10.4f} {row[8]:<10.4f}")
        
    print("\n--- SCORECARD B: OOD CALIBRATION GENERALIZATION (DATASET B) ---")
    print(f"{'Classifier':<15} {'F1-Score':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'ROC-AUC':<10} {'PR-AUC':<10} {'FPR':<10} {'FNR':<10}")
    for row in scorecard_cal:
        print(f"{row[0]:<15} {row[1]:<10.4f} {row[2]:<10.4f} {row[3]:<10.4f} {row[4]:<10.4f} {row[5]:<10.4f} {row[6]:<10.4f} {row[7]:<10.4f} {row[8]:<10.4f}")
    
    # Select Winner based on Dataset B F1-score (OOD robustness)
    best_f1 = -1.0
    winner_name = None
    for row in scorecard_cal:
        if row[1] > best_f1:
            best_f1 = row[1]
            winner_name = row[0].lower()
            
    print(f"\n[+] Production Model Selection Choice: {winner_name.upper()} (F1-score: {best_f1:.4f})")
    
    # Save winner model
    winner_pipeline = candidates[winner_name]
    joblib.dump(winner_pipeline, 'model.joblib')
    print(f"[+] Saved winning classifier package to: 'model.joblib'")
    print("[+] Step 3 finished successfully.")

if __name__ == '__main__':
    main()
