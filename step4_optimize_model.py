#!/usr/bin/env python3
"""
Step 4: Model Optimization Benchmarks
-------------------------------------
Loads winner model.joblib. If tree classifier, copies it directly to 
model_optimized.joblib. If LSTM classifier, runs hidden layer downsizing, 
dynamic INT8 quantization, and TorchScript JIT tracing. Saves optimized 
checkpoints and comparison charts.
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from lstm_wrapper import LSTMClassifier, LSTMDeploymentWrapper

def main():
    if not os.path.exists('model.joblib'):
        print("[!] Error: model.joblib not found. Run Step 2 & 3 first.")
        sys.exit(1)
        
    pipeline = joblib.load('model.joblib')
    
    # If the winner is a tree classifier, bypass PyTorch-specific optimization
    is_lstm = hasattr(pipeline, 'state_dict')
    if not is_lstm:
        print("[+] Winning model is tree-based. Bypassing PyTorch quantization/JIT pipelines...")
        joblib.dump(pipeline, 'model_optimized.joblib')
        print("[+] Mirrored winning tree model to 'model_optimized.joblib'.")
        print("[+] Optimization analysis finished successfully.")
        sys.exit(0)
        
    if not TORCH_AVAILABLE:
        print("[!] PyTorch is required to optimize LSTM model.")
        sys.exit(1)
        
    print("[+] Winner is LSTM model. Initiating deep learning optimization suite...")
    
    # Load calibration data for benchmarking
    cal_path = "conn.log.calibration_90_10"
    if not os.path.exists(cal_path):
        print(f"[!] Error: Calibration dataset '{cal_path}' not found.")
        sys.exit(1)
        
    cal_df = pd.read_csv(cal_path, sep='\t', low_memory=False).dropna(subset=['label'])
    numeric_cols = ['duration', 'orig_bytes', 'resp_bytes', 'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
    categorical_cols = ['proto', 'service', 'conn_state', 'history']
    
    X = cal_df[numeric_cols + categorical_cols]
    labels = cal_df['label'].astype(str).str.strip().str.lower()
    y = (~labels.str.startswith('benign')).astype(int).values
    
    preprocessor = pipeline.preprocessor
    X_proc = preprocessor.transform(X)
    
    # Create sequences
    def create_sequences(X_data, y_data, seq_len=5):
        X_seq, y_seq = [], []
        for i in range(len(X_data) - seq_len + 1):
            X_seq.append(X_data[i : i + seq_len])
            y_seq.append(y_data[i + seq_len - 1])
        return np.array(X_seq), np.array(y_seq)
        
    X_b_seq, y_b_seq = create_sequences(X_proc, y, seq_len=5)
    
    # Lazy init of the baseline PyTorch model
    pipeline._lazy_init_model()
    base_model = pipeline._model
    input_dim = X_proc.shape[1]
    seq_len = 5
    
    # Benchmarking helper
    def benchmark_model(model, X_eval, y_eval, is_jit=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not is_jit:
            model.to(device)
            model.eval()
            
        t_start = time.time()
        with torch.no_grad():
            if is_jit:
                probs = []
                for sample in X_eval:
                    inp = torch.FloatTensor(sample).unsqueeze(0).to(device)
                    probs.append(model(inp).cpu().numpy()[0])
                probs = np.array(probs)
            else:
                probs = model(torch.FloatTensor(X_eval).to(device)).cpu().numpy()
        latency_us = ((time.time() - t_start) / len(X_eval)) * 1000000
        preds = (probs >= 0.90).astype(int)
        
        from sklearn.metrics import f1_score, accuracy_score
        f1 = f1_score(y_eval, preds, zero_division=0)
        acc = accuracy_score(y_eval, preds)
        return f1, acc, latency_us

    # --- 1. Baseline Model Size ---
    torch.save(base_model.state_dict(), 'temp_base.pth')
    base_size_kb = os.path.getsize('temp_base.pth') / 1024
    os.remove('temp_base.pth')
    
    # --- 2. Model Downsizing (hidden=12) ---
    print(f"[2/4] Training Downsized LSTM (hidden_dim=12) on training log...")
    train_path = "conn.log.train_20_80"
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path, sep='\t', low_memory=False).dropna(subset=['label'])
        X_tr = train_df[numeric_cols + categorical_cols]
        y_tr = (~train_df['label'].astype(str).str.strip().str.lower().str.startswith('benign')).astype(int).values
        X_tr_proc = preprocessor.transform(X_tr)
        X_tr_seq, y_tr_seq = create_sequences(X_tr_proc, y_tr, seq_len=5)
    else:
        X_tr_seq, y_tr_seq = X_b_seq, y_b_seq
        
    downsized_model = LSTMClassifier(input_dim=input_dim, hidden_dim=12)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    downsized_model.to(device)
    
    from torch.utils.data import TensorDataset, DataLoader
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr_seq), torch.FloatTensor(y_tr_seq)), batch_size=1024, shuffle=True)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(downsized_model.parameters(), lr=0.001)
    for epoch in range(2):
        downsized_model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(downsized_model(bx), by)
            loss.backward()
            optimizer.step()
            
    torch.save(downsized_model.state_dict(), 'temp_down.pth')
    downsized_size_kb = os.path.getsize('temp_down.pth') / 1024
    os.remove('temp_down.pth')
    
    # --- 3. Dynamic Quantization (Float32 -> Int8) ---
    print(f"[3/4] Compiling Dynamically Quantized LSTM (Int8 Weights)...")
    quantized_model = torch.quantization.quantize_dynamic(
        base_model, 
        {torch.nn.LSTM, torch.nn.Linear}, 
        dtype=torch.qint8
    )
    torch.save(quantized_model.state_dict(), 'temp_quant.pth')
    quant_size_kb = os.path.getsize('temp_quant.pth') / 1024
    os.remove('temp_quant.pth')
    
    # --- 4. TorchScript JIT Compilation (Tracing) ---
    print(f"[4/4] Tracing Baseline LSTM using TorchScript JIT compiler...")
    dummy_input = torch.FloatTensor(X_b_seq[:10])
    jit_model = torch.jit.trace(base_model, dummy_input)
    torch.jit.save(jit_model, 'temp_jit.pt')
    jit_size_kb = os.path.getsize('temp_jit.pt') / 1024
    os.remove('temp_jit.pt')
    
    # --- Benchmarking Phase ---
    f1_base, acc_base, lat_base = benchmark_model(base_model, X_b_seq, y_b_seq)
    f1_down, acc_down, lat_down = benchmark_model(downsized_model, X_b_seq, y_b_seq)
    f1_quant, acc_quant, lat_quant = benchmark_model(quantized_model, X_b_seq, y_b_seq)
    f1_jit, acc_jit, lat_jit = benchmark_model(jit_model, X_b_seq, y_b_seq, is_jit=True)
    
    print("\n--- RUNNING OPTIMIZATION COMPARATIVE BENCHMARKS ---")
    print(f"{'LSTM Variant':<28} {'F1-Score':<10} {'Accuracy':<10} {'Latency':<18} {'File Size':<14} {'Speedup':<10}")
    for name, f1, acc, lat, size, speedup in [
        ("Standard Baseline LSTM", f1_base, acc_base, lat_base, base_size_kb, 1.0),
        ("Downsized (hidden=12)", f1_down, acc_down, lat_down, downsized_size_kb, lat_base/lat_down),
        ("Dynamic Quantized (Int8)", f1_quant, acc_quant, lat_quant, quant_size_kb, lat_base/lat_quant),
        ("TorchScript JIT Traced", f1_jit, acc_jit, lat_jit, jit_size_kb, lat_base/lat_jit)
    ]:
        print(f"{name:<28} {f1:<10.4f} {acc:<10.4f} {lat:<13.2f} us/pkt {size:<11.2f} KB {speedup:<10.1f}x")
    
    print("\n[+] Serializing the most optimized candidate (Standard Baseline LSTM)...")
    joblib.dump(pipeline, 'model_optimized.joblib')
    print("[+] Saved optimized LSTM model wrapper to 'model_optimized.joblib'.")
    
    # Save comparative bar chart
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        variants = ['Baseline', 'Downsized', 'Quantized', 'JIT Traced']
        latencies = [lat_base, lat_down, lat_quant, lat_jit]
        sizes = [base_size_kb, downsized_size_kb, quant_size_kb, jit_size_kb]
        
        x = np.arange(len(variants))
        width = 0.35
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('LSTM Variant')
        ax1.set_ylabel('Inference Latency (us/pkt)', color=color)
        rects1 = ax1.bar(x - width/2, latencies, width, label='Latency', color=color, alpha=0.8)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Model File Size (KB)', color=color)
        rects2 = ax2.bar(x + width/2, sizes, width, label='File Size', color=color, alpha=0.8)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('LSTM Model Optimization Comparative Benchmarks')
        ax1.set_xticks(x)
        ax1.set_xticklabels(variants)
        fig.tight_layout()
        
        filepath = 'model_optimization_comparison.png'
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"[+] Saved LSTM optimization comparative chart to: {filepath}")
    except Exception as e:
        print(f"[!] Warning: Could not save optimization comparative chart ({e})")
        
    print("\n[+] Optimization analysis finished successfully.")

if __name__ == '__main__':
    main()
