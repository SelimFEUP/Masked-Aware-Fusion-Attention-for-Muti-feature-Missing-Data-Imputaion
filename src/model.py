import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Conv1D, BatchNormalization, ReLU, Add, Input, Lambda, MultiHeadAttention, Concatenate, LayerNormalization

def residual_tcn_block(input_layer, filters=64, kernel_size=3):
    conv1 = Conv1D(filters, kernel_size, padding='same')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)
    conv2 = Conv1D(filters, kernel_size, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)
    
    # Project input if needed
    if input_layer.shape[-1] != filters:
        input_layer = Conv1D(filters, kernel_size=1, padding='same')(input_layer)
    
    res = Add()([input_layer, conv2])  # Residual connection
    return res

class TemporalGap(tf.keras.layers.Layer):
    def call(self, mask):
        inv = 1.0 - mask  # 1 when missing
        return tf.cumsum(inv, axis=1)

class TemporalDecay(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.w = self.add_weight(shape=(1,), initializer="ones")

    def call(self, delta_t):
        return tf.exp(-tf.nn.relu(self.w) * delta_t)

def build_imputation_model(input_shape, num_features):
    inp = Input(shape=input_shape)
    values = inp[..., :num_features]
    masks  = inp[..., num_features:]

    delta_t = TemporalGap()(masks)
    gamma   = TemporalDecay()(delta_t)
    values  = values * gamma

    x = residual_tcn_block(values, 64)
    x = residual_tcn_block(x, 64)
    
    attn_mask = Lambda(lambda m: tf.expand_dims(m[...,0], axis=1))(masks)


    attn = MultiHeadAttention(num_heads=8, key_dim=32)(
            query=x, value=x, key=x, attention_mask=attn_mask
       )

    fusion = Concatenate()([x, attn])
    gate = Dense(64, activation="sigmoid")(fusion)
    x = LayerNormalization()(x + gate * attn)

    x = tf.keras.layers.Bidirectional(LSTM(64, return_sequences=True))(x)

    mu = Dense(num_features)(x)
    return Model(inp, mu)
