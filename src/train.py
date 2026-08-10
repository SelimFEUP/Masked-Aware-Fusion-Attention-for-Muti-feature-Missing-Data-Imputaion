import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Conv1D, BatchNormalization, ReLU, Add, Input, Lambda, MultiHeadAttention, Concatenate, LayerNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from src.model import build_imputation_model
from src.preprocessing import *

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

physical_adj = load_adj_matrix("bay_adj_mx.pkl")

class MaskedMAE(tf.keras.losses.Loss):
    def __init__(self, F, lambda_obs=0.3):
        super().__init__()
        self.F = F
        self.lambda_obs = lambda_obs

    def call(self, y_true, y_pred):
        values = y_true[..., :self.F]
        mask = y_true[..., self.F:]

        valid_locations = tf.cast(~tf.math.is_nan(values), tf.float32)
        safe_values = tf.where(tf.math.is_nan(values), tf.zeros_like(values), values)

        missing_mask = (1.0 - mask) * valid_locations
        observed_mask = mask * valid_locations

        abs_error = tf.abs(y_pred - safe_values)
        missing_loss = tf.reduce_sum(abs_error * missing_mask) / (tf.reduce_sum(missing_mask) + 1e-8)
        observed_loss = tf.reduce_sum(abs_error * observed_mask) / (tf.reduce_sum(observed_mask) + 1e-8)

        return missing_loss + self.lambda_obs * observed_loss

def train_model():
  model = build_imputation_model(X_train.shape[1:], F, predefined_adj=physical_adj, dropout_rate=0.15, l2_reg=1e-4)
  optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=1e-3, weight_decay=1e-5)
  model.compile(optimizer=optimizer, loss=MaskedMAE(F, lambda_obs=0.8))
  model.summary()
  callbacks = [
    ModelCheckpoint('pems_random.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1),
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6)
   ]
  history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=150, batch_size=48, callbacks=callbacks)
  return history
