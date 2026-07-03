import numpy as np
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Last time step
        return self.sigmoid(self.fc(out)).squeeze(-1)

class LSTMDeploymentWrapper:
    def __init__(self, state_dict, input_dim, preprocessor, seq_len=5, hidden_dim=32):
        self.state_dict = state_dict
        self.input_dim = input_dim
        self.preprocessor = preprocessor
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self._model = None
        
    def _lazy_init_model(self):
        if self._model is None:
            self._model = LSTMClassifier(self.input_dim, hidden_dim=self.hidden_dim)
            self._model.load_state_dict(self.state_dict)
            self._model.eval()
            
    def predict_proba(self, X):
        self._lazy_init_model()
        import pandas as pd
        
        # 1. Apply preprocessor if X is raw DataFrame
        if isinstance(X, pd.DataFrame):
            X_proc = self.preprocessor.transform(X)
        else:
            X_proc = X
            
        # 2. Extract sequences dynamically for predict-time formatting
        n_samples = len(X_proc)
        sequences = []
        for i in range(n_samples):
            start_idx = max(0, i - self.seq_len + 1)
            seq = X_proc[start_idx:i+1]
            if len(seq) < self.seq_len:
                pad_width = self.seq_len - len(seq)
                seq = np.pad(seq, ((pad_width, 0), (0, 0)), mode='constant')
            sequences.append(seq)
            
        X_seq = np.array(sequences)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        
        with torch.no_grad():
            probs = self._model(X_tensor).numpy().ravel()
            
        return np.column_stack((1 - probs, probs))
        
    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.90).astype(int)
