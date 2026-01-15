import tensorflow as tf
import pandas as pd
import numpy as np
from src.train import train_model
from src.evaluate import *
from src.preprocessing import *

def main():
    # Load the model
    model.load_weights('models/pems_random.keras')
  
    # ---------- Inference ----------
    preds = model.predict(X_test, batch_size=32)
    imputed_scaled = reconstruct_from_windows(preds, len(test_raw), window)
    imputed_scaled[test_mask == 1] = test_scaled[test_mask == 1]
    imputed = scaler.inverse_transform(imputed_scaled)
    
    # Evaluate the model on the test set
    rmse, mae = compute_errors(imputed, test_raw, test_eval_mask.values)
    mape = compute_smape(imputed, test_raw, test_eval_mask.values)
    print(f"\nFINAL → RMSE: {rmse:.4f}   MAE: {mae:.4f}  MAPE: {mape:.4f}")
    
if __name__ == "__main__":
    main()
    
