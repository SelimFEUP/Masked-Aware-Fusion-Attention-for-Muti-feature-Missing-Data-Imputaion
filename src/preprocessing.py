import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def fiber_missing_fn(df, fiber_length=12, seed=42):
    rng = np.random.default_rng(seed)

    # mask=True means keep value
    mask = np.ones(df.shape, dtype=bool)

    for col in range(df.shape[1]):
        if df.shape[0] <= fiber_length:
            start = 0
        else:
            start = rng.integers(0, df.shape[0] - fiber_length)

        mask[start:start + fiber_length, col] = False  # False = missing region

    corrupted = df.copy()
    corrupted[~mask] = np.nan

    eval_mask = ~mask
    return corrupted, pd.DataFrame(eval_mask, index=df.index, columns=df.columns)

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

def create_windows(values_filled, values_clean, masks, window=24):
    X, y = [], []
    for i in range(len(values_filled) - window):
        X.append(np.concatenate([values_filled[i:i+window],
                                 masks[i:i+window]], axis=-1))
        y.append(np.concatenate([values_clean[i:i+window],
                                 masks[i:i+window]], axis=-1))
    return np.array(X), np.array(y)



def reconstruct_from_windows(preds, T, window):
    F = preds.shape[2]
    series = np.zeros((T, F))
    counts = np.zeros((T, F))
    for i in range(len(preds)):
        series[i:i+window] += preds[i]
        counts[i:i+window] += 1
    return series / np.maximum(counts, 1)

df = pd.read_csv("data/PEMS_BAY.csv", index_col=0)
values = df.values
T, F = values.shape
window = 24

split = int(T * 0.8)
train_raw = values[:split]
test_raw  = values[split:]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_raw)
test_scaled  = scaler.transform(test_raw)

train_missing, train_eval_mask = random_missing_fn(pd.DataFrame(train_scaled), 0.20, 7)
test_missing,  test_eval_mask  = random_missing_fn(pd.DataFrame(test_scaled),  0.20, 7)

train_mask = (~train_missing.isna()).astype(float).values
test_mask  = (~test_missing.isna()).astype(float).values

train_filled = train_missing.fillna(train_missing.mean()).values
test_filled  = test_missing.fillna(test_missing.mean()).values

val_split = int(len(train_filled) * 0.9)

train_filled_tr = train_filled[:val_split]
train_clean_tr  = train_scaled[:val_split]
train_mask_tr   = train_mask[:val_split]

val_filled_tr = train_filled[val_split - window:]
val_clean_tr  = train_scaled[val_split - window:]
val_mask_tr   = train_mask[val_split - window:]

X_train, y_train = create_windows(train_filled_tr, train_clean_tr, train_mask_tr, window)
X_val,   y_val   = create_windows(val_filled_tr,  val_clean_tr,  val_mask_tr,  window)

X_test,  y_test  = create_windows(test_filled, test_scaled, test_mask, window)
