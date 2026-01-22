import numpy as np

def compute_errors(imputed, truth, eval_mask):
    diff = imputed[eval_mask] - truth[eval_mask]
    rmse = np.sqrt(np.mean(diff**2))
    mae  = np.mean(np.abs(diff))
    return rmse, mae

# Compute safe sMAPE only at positions indicated by eval_mask, ignoring near-zero values and NaNs to prevent explosion.  
def compute_smape(imputed, truth, eval_mask, eps=1e-3):
        
    y_true = truth[eval_mask]
    y_pred = imputed[eval_mask]

    # Safe mask: ignore near-zero values and NaNs
    mask = (np.abs(y_true) > eps) & ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(mask) == 0:
        return np.nan

    y_true_safe = y_true[mask]
    y_pred_safe = y_pred[mask]

    denom = np.maximum(np.abs(y_true_safe) + np.abs(y_pred_safe), eps)
    smape = np.mean(2 * np.abs(y_pred_safe - y_true_safe) / denom) * 100
    return smape
