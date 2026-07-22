#!/usr/bin/env python3
# Step 5: Evaluate the optimized model on test and calibration sets and generate confusion matrices.

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
    cal_path = "conn.log.calibration_60_40"
    
    if not os.path.exists(test_path) or not os.path.exists(cal_path):
        print("[!] Error: test/calibration splits not found.")
        sys.exit(1)
        
    if not os.path.exists('model_optimized.joblib'):
        print("[!] Error: model_optimized.joblib not found. Run Step 4 first.")
        sys.exit(1)
        
    pipeline = joblib.load('model_optimized.joblib')
    
    print("[+] Loading test and OOD calibration splits...")
    test_df = pd.read_csv(test_path, sep='\t', low_memory=False).dropna(subset=['label'])
    cal_df = pd.read_csv(cal_path, sep='\t', low_memory=False).dropna(subset=['label'])
    
    numeric_cols = ['duration', 'orig_bytes', 'resp_bytes', 'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
    categorical_cols = ['proto', 'service', 'conn_state', 'history']
    
    def get_xy(df):
        X = df[numeric_cols + categorical_cols].copy()
        labels = df['label'].astype(str).str.strip().str.lower()
        y = (~labels.str.startswith('benign')).astype(int).values
        return X, y
        
    X_test, y_test = get_xy(test_df)
    X_cal, y_cal = get_xy(cal_df)
    
    is_lstm = hasattr(pipeline, 'state_dict')
    
    # Evaluate on Dataset A (Test set)
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
        
    acc_a, prec_a, rec_a, f1_a, roc_auc_a, pr_auc_a, fpr_a, fnr_a = evaluate_predictions(y_eval, preds, probs)
    
    # Save confusion matrix for Dataset A
    plt.figure(figsize=(5, 5))
    plt.imshow(confusion_matrix(y_eval, preds, labels=[0, 1]), interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Optimized Model Dataset A confusion matrix')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('confusion_matrix_optimized_lstm_dataset_a.png')
    plt.close()
    
    # Evaluate on Dataset B (Calibration set)
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
        
    acc_b, prec_b, rec_b, f1_b, roc_auc_b, pr_auc_b, fpr_b, fnr_b = evaluate_predictions(y_eval_cal, preds_cal, probs_cal)
    
    # Save confusion matrix for Dataset B
    plt.figure(figsize=(5, 5))
    plt.imshow(confusion_matrix(y_eval_cal, preds_cal, labels=[0, 1]), interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title('Optimized Model Dataset B confusion matrix')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('confusion_matrix_optimized_lstm_dataset_b.png')
    plt.close()
    
    print("\n--- OPTIMIZED MODEL PERFORMANCE SCORECARD ---")
    print(f"{'Test Domain':<28} {'F1-Score':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'ROC-AUC':<10} {'PR-AUC':<10} {'FPR':<10} {'FNR':<10}")
    for name, f1, acc, prec, rec, roc, pr, fpr, fnr in [
        ("Dataset A (Temporal Test)", f1_a, acc_a, prec_a, rec_a, roc_auc_a, pr_auc_a, fpr_a, fnr_a),
        ("Dataset B (OOD Calib Log)", f1_b, acc_b, prec_b, rec_b, roc_auc_b, pr_auc_b, fpr_b, fnr_b)
    ]:
        print(f"{name:<28} {f1:<10.4f} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {roc:<10.4f} {pr:<10.4f} {fpr:<10.4f} {fnr:<10.4f}")
    print("[+] Step 5 finished successfully.")

if __name__ == '__main__':
    main()

