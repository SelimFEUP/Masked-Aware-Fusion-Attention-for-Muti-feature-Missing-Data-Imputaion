import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


def load_adj_matrix(adj_path="bay_adj_mx.pkl"):
    with open(adj_path, 'rb') as f:
        adj = pickle.load(f)
    # Usually it's a dict with 'adj_mat' or similar
    if isinstance(adj, dict):
        adj = adj['adj_mat'] if 'adj_mat' in adj else adj[list(adj.keys())[0]]
    adj = np.array(adj, dtype=np.float32)
    # Normalize adjacency (common practice)
    adj = adj / (adj.sum(axis=1, keepdims=True) + 1e-8)
    print(f"Loaded adjacency matrix: {adj.shape}")
    return tf.convert_to_tensor(adj, dtype=tf.float32)

def create_windows(values_filled, values_clean, masks, window=76, stride=1):
    X, y = [], []
    for i in range(0, len(values_filled) - window + 1, stride):
        X.append(np.concatenate([values_filled[i:i+window], masks[i:i+window]], axis=-1))
        y.append(np.concatenate([values_clean[i:i+window], masks[i:i+window]], axis=-1))
    return np.array(X), np.array(y)

def reconstruct_from_windows_weighted(preds, T, window, F=207, stride=1, window_type="hanning"):
    """
    Reconstructs full time series using Smooth Overlap-Add (OLA) with window weighting.
    
    Parameters:
    - preds: np.ndarray of shape (N, window, F)
    - T: int, total length of the original time series
    - window: int, window size (e.g., 76)
    - F: int, number of features/nodes (e.g., 207)
    - stride: int, stride used during window creation
    - window_type: str, 'hanning', 'gaussian', or 'triang'
    
    Returns:
    - Reconstructed time series array of shape (T, F)
    """
    series = np.zeros((T, F), dtype=np.float32)
    weight_sum = np.zeros((T, F), dtype=np.float32)
    
    # 1. Generate 1D temporal weighting curve
    if window_type == "hanning":
        # Small offset (1e-2) prevents pure zero weights at sequence endpoints
        w = np.hanning(window) + 1e-2
    elif window_type == "gaussian":
        sigma = window / 4.0
        x = np.arange(window) - (window - 1) / 2.0
        w = np.exp(-0.5 * (x / sigma) ** 2)
    elif window_type == "triang":
        w = np.triang(window) + 1e-2
    else:
        w = np.ones(window)

    # Broadcast to match feature dimensions: shape (window, 1)
    w_expanded = w[:, np.newaxis]

    # 2. Accumulate weighted predictions
    for i, pred in enumerate(preds):
        start = i * stride
        end = start + window
        
        if end <= T:
            series[start:end] += pred * w_expanded
            weight_sum[start:end] += w_expanded
        else:
            # Handle tail edge cases where window exceeds total length T
            valid_len = T - start
            series[start:T] += pred[:valid_len] * w_expanded[:valid_len]
            weight_sum[start:T] += w_expanded[:valid_len]

    # 3. Normalize by total accumulated weight
    return series / np.maximum(weight_sum, 1e-8)

def random_missing_fn(df, p=0.2, block=6, seed=7):
    rng = np.random.default_rng(seed)
    mask = np.ones(df.shape, dtype=bool)

    for col in range(df.shape[1]):
        i = 0
        while i < df.shape[0]:
            if rng.random() < p:
                mask[i:i+block, col] = False
                i += block
            else:
                i += 1

    corrupted = df.copy()
    corrupted[~mask] = np.nan
    return corrupted, pd.DataFrame(~mask, index=df.index, columns=df.columns)

df = pd.read_csv("PEMS_BAY.csv", index_col=0)
values = df.values
T, F = values.shape
window = 24

split = int(T * 0.8)
train_raw, test_raw = values[:split], values[split:]

# Switch back to StandardScaler
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_raw)
test_scaled  = scaler.transform(test_raw)

# Inject missingness (same as before)
train_missing, train_eval_mask = random_missing_fn(pd.DataFrame(train_scaled), 0.20, 7)
test_missing,  test_eval_mask  = random_missing_fn(pd.DataFrame(test_scaled),  0.20, 7)

train_mask = (~train_missing.isna()).astype(float).values
test_mask  = (~test_missing.isna()).astype(float).values

# Interpolation is critical now because our model explicitly uses it as a baseline skip-connection
train_filled = train_missing.interpolate(limit_direction='both').fillna(0).values
test_filled  = test_missing.interpolate(limit_direction='both').fillna(0).values

val_split_idx = int(len(train_filled) * 0.9)
train_filled_tr, val_filled_tr = train_filled[:val_split_idx], train_filled[val_split_idx:]
train_clean_tr, val_clean_tr   = train_scaled[:val_split_idx], train_scaled[val_split_idx:]
train_mask_tr, val_mask_tr     = train_mask[:val_split_idx], train_mask[val_split_idx:]

# Keep Stride=3 for training to prevent overfitting on overlapping sequences
X_train, y_train = create_windows(train_filled_tr, train_clean_tr, train_mask_tr, window, stride=3)
X_val,   y_val   = create_windows(val_filled_tr, val_clean_tr, val_mask_tr, window, stride=3)
X_test,  y_test  = create_windows(test_filled, test_scaled, test_mask, window, stride=1) 
