import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Conv1D, BatchNormalization, ReLU, Add, Input, Lambda, MultiHeadAttention, Concatenate, LayerNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.model import build_imputation_model
from src.preprocessing import *

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

class MaskedMSE(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        values = y_true[..., :F]
        mask   = y_true[..., F:]

        mse_all = tf.reduce_mean(tf.square(y_pred - values))
        mse_miss = tf.reduce_sum(tf.square((y_pred - values) * (1-mask))) / (tf.reduce_sum(1-mask)+1e-8)

        return 0.1 * mse_all + mse_miss

def train_model():
  model = build_imputation_model(X_train.shape[1:], F)
  model.compile(optimizer='adam', loss=MaskedMSE())
  model.summary()
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=15, restore_best_weights=True)
  mc = tf.keras.callbacks.ModelCheckpoint('models/pems_random.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
  history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, validation_split=0.1, callbacks=[mc, early_stopping])
  return history
