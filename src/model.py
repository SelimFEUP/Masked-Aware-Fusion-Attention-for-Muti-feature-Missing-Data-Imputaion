
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Conv1D, BatchNormalization, ReLU, Add, Input, Lambda, MultiHeadAttention, Concatenate, LayerNormalization

def residual_tcn_block(x, filters=64, kernel_size=3, dilation_rate=1):
    shortcut = x

    x = Conv1D(filters, kernel_size, padding='same',
               dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv1D(filters, kernel_size, padding='same',
               dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)

    x = Add()([shortcut, x])
    return ReLU()(x)

class TemporalGap(tf.keras.layers.Layer):
    def call(self, mask):
        inv = 1.0 - mask  # 1 when missing
        return tf.cumsum(inv, axis=1)

class TemporalDecay(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.w = self.add_weight(shape=(1,), initializer="ones")
        self.alpha = self.add_weight(shape=(1,), initializer="zeros")

    def call(self, delta_t, A):
        #neighbor_delta = tf.einsum("ij,btj->bti", A, delta_t)
        neighbor_delta = tf.einsum("bij,btj->bti", A, delta_t)
        delta_hat = delta_t + self.alpha * neighbor_delta
        return tf.exp(-tf.nn.relu(self.w) * delta_hat)

class DynamicGraph(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.q = Dense(d_model)
        self.k = Dense(d_model)

    def call(self, x):
        # x: (B, T, N)
        x = tf.transpose(x, [0, 2, 1])
        q = self.q(x)
        k = self.k(x)
        
        attn = tf.matmul(q, k, transpose_b=True)
        attn /= tf.sqrt(tf.cast(tf.shape(q)[-1], tf.float32))

        return tf.nn.softmax(attn, axis=-1)  # (B, N, N)


class MaskAwareAttention(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.attn = MultiHeadAttention(num_heads=4, key_dim=d_model)

    def call(self, x, mask):
        # mask: (B, T, 1) -> (B, T)
        attn_mask = tf.cast(mask[:, :, 0], tf.int32)

        return self.attn(
            query=x,
            value=x,
            key=x,
            attention_mask=attn_mask[:, tf.newaxis, :]
        )

                    
def build_imputation_model(input_shape, num_features):
    inp = Input(shape=input_shape)
    values = inp[..., :num_features]
    masks  = inp[..., num_features:]

    delta_t = TemporalGap()(masks)
    #A = tf.constant(A, dtype=tf.float32)
    A_layer = DynamicGraph(num_features)
    A_matrix = A_layer(values)
    
    gamma   = TemporalDecay()(delta_t, A_matrix)
    values  = values * gamma

    x = residual_tcn_block(values, 64)
    x = residual_tcn_block(x, 64)
    #x = residual_tcn_block(x, 64, dilation_rate=3)
    
    attn_mask = Lambda(lambda m: tf.expand_dims(m[...,0], axis=-1))(masks)


    attn = MaskAwareAttention(128)(x, attn_mask)

    fusion = Concatenate()([x, attn])
    gate = Dense(64, activation="sigmoid")(fusion)
    x = LayerNormalization()(x + gate * attn)
    
    h_val = Dense(64, activation="relu")(x)
    h_mask = Dense(64, activation="relu")(masks)

    gamma_m = Dense(64)(h_mask)
    beta_m  = Dense(64)(h_mask)

    x = LayerNormalization()(gamma_m * h_val + beta_m)

    mu = Dense(num_features)(x)
    return Model(inp, mu)
