import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Conv1D, SeparableConv1D, LayerNormalization, 
    Add, Multiply, Activation, SpatialDropout1D, 
    MultiHeadAttention, Input, Lambda, Concatenate
)
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len=5000, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len

    def build(self, input_shape):
        self.d_model = input_shape[-1]
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        if self.d_model % 2 != 0:
            pe[:, 1::2] = np.cos(position * div_term)[:, :-1]
        else:
            pe[:, 1::2] = np.cos(position * div_term)
            
        self.pe = tf.constant(pe[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len})
        return config


def gated_multiscale_mixer(
    x, filters=128, num_heads=4, dropout_rate=0.15, l2_reg=1e-5, name_prefix="gmstm"
):
    reg = tf.keras.regularizers.l2(l2_reg)

    if int(x.shape[-1]) != filters:
        x = Conv1D(filters, 1, padding="same", kernel_regularizer=reg, name=f"{name_prefix}_proj")(x)
    shortcut = x

    # Path A: Multi-Scale Dilated Depthwise Convolutions
    x_n = LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln_conv")(x)

    conv1 = SeparableConv1D(filters, 3, padding="same", dilation_rate=1,
                            depthwise_regularizer=reg, pointwise_regularizer=reg)(x_n)
    conv2 = SeparableConv1D(filters, 3, padding="same", dilation_rate=2,
                            depthwise_regularizer=reg, pointwise_regularizer=reg)(x_n)
    conv3 = SeparableConv1D(filters, 3, padding="same", dilation_rate=4,
                            depthwise_regularizer=reg, pointwise_regularizer=reg)(x_n)

    multi_scale = Add()([conv1, conv2, conv3])
    multi_scale = Activation("gelu")(multi_scale)

    # Path B: Global Temporal Attention
    x_n2 = LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln_attn")(x)
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=filters // num_heads, dropout=dropout_rate)(x_n2, x_n2)

    # Fusion + SwiGLU Gating
    fused = Concatenate()([multi_scale, attn])
    fused = Conv1D(filters * 2, 1, padding="same", kernel_regularizer=reg)(fused)

    a, b = tf.split(fused, 2, axis=-1)
    gated = Multiply()([a, Activation("swish")(b)])
    gated = SpatialDropout1D(dropout_rate)(gated)

    out = Add(name=f"{name_prefix}_out")([shortcut, gated])
    return out


class TemporalGap(tf.keras.layers.Layer):
    def call(self, mask):
        mask = tf.cast(mask, tf.float32)
        def step(prev_gap, current_mask):
            return tf.where(current_mask > 0.5, tf.zeros_like(prev_gap), prev_gap + 1.0)
        
        # Forward
        gaps_fwd = tf.scan(step, elems=tf.transpose(mask, [1, 0, 2]), initializer=tf.zeros_like(mask[:, 0, :]))
        gaps_fwd = tf.transpose(gaps_fwd, [1, 0, 2])
        
        # Backward
        mask_rev = mask[:, ::-1, :]
        gaps_bwd = tf.scan(step, elems=tf.transpose(mask_rev, [1, 0, 2]), initializer=tf.zeros_like(mask[:, 0, :]))
        gaps_bwd = tf.transpose(gaps_bwd, [1, 0, 2])[:, ::-1, :]
        
        # Stack into (Batch, Time, Nodes, 2)
        return tf.stack([tf.math.log(gaps_fwd + 1.0), tf.math.log(gaps_bwd + 1.0)], axis=-1)


class DynamicGraph(tf.keras.layers.Layer):
    def __init__(self, num_nodes, predefined_adj=None, d_model=64, **kwargs):
        super().__init__(**kwargs)
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.predefined_adj = predefined_adj
        
    def build(self, input_shape):
        self.node_emb = self.add_weight(
            shape=(self.num_nodes, self.d_model), 
            initializer="glorot_uniform", 
            trainable=True, name="node_emb"
        )
        
        # To inject the physical matrix
        if self.predefined_adj is not None:
            self.static_adj = tf.constant(self.predefined_adj, dtype=tf.float32)
        else:
            self.static_adj = self.add_weight(
                shape=(self.num_nodes, self.num_nodes), 
                initializer="glorot_uniform", 
                trainable=True, name="static_adj"
            )
            
        self.lambda_graph = self.add_weight(
            shape=(1,), 
            initializer=tf.keras.initializers.Constant(0.7), 
            trainable=True, name="lambda_graph"
        )
        super().build(input_shape)

    def call(self, inputs=None):
        scores = tf.matmul(self.node_emb, self.node_emb, transpose_b=True) / tf.sqrt(tf.cast(self.d_model, tf.float32))
        dynamic_adj = tf.nn.softmax(scores, axis=-1)
        
        # If using physical adj, we might not need softmax on it if it's already normalized,
        # but applying it keeps the scales consistent.
        static_adj = tf.nn.softmax(self.static_adj, axis=-1) 
        
        lam = tf.sigmoid(self.lambda_graph)
        return lam * static_adj + (1 - lam) * dynamic_adj

    def get_config(self):
        config = super().get_config()
        config.update({"num_nodes": self.num_nodes, "d_model": self.d_model})
        return config


def build_imputation_model(input_shape, num_features, predefined_adj=None, dropout_rate=0.15, l2_reg=1e-4):
    reg = tf.keras.regularizers.l2(l2_reg)
    inp = Input(shape=input_shape)

    values = inp[..., :num_features]
    masks  = inp[..., num_features:]

    # 1. Base spatial aggregation (Mask-aware)
    A_matrix = DynamicGraph(num_features, predefined_adj=predefined_adj, name="dynamic_graph")(values)
    
    def mask_aware_spatial(args):
        A, vals, m = args
        obs_vals = vals * m
        num = tf.einsum("ij,btj->bti", A, obs_vals)
        den = tf.einsum("ij,btj->bti", A, m)
        return num / (den + 1e-4)
        
    spatial_base = Lambda(mask_aware_spatial, name="spatial_aggregation")([A_matrix, values, masks])

    # 2. Extract Temporal Gaps
    delta_t = TemporalGap(name="temporal_gap")(masks)

    # 3. Project to 4D Tensor
    v_ex = tf.expand_dims(values, -1)
    m_ex = tf.expand_dims(masks, -1)
    s_ex = tf.expand_dims(spatial_base, -1)
    
    # Concatenate features
    x = Concatenate(axis=-1)([v_ex, m_ex, s_ex, delta_t])
    
    d_model = 64
    x = Dense(d_model, kernel_regularizer=reg, name="feature_proj")(x)

    # 4. Alternating Spatio-Temporal Blocks
    for i in range(2):
        # A. TEMPORAL MIXING (Node-independent)
        b_shape = tf.shape(x)[0]
        t_shape = tf.shape(x)[1]
        n_shape = tf.shape(x)[2]
        
        # Reshape to (Batch * Nodes, Time, Channels) for the 1D mixer
        x_temp = tf.transpose(x, [0, 2, 1, 3]) # (B, N, T, C)
        x_temp = tf.reshape(x_temp, [-1, t_shape, d_model])
        
        if i == 0:
            x_temp = PositionalEncoding(name=f"pos_enc_{i}")(x_temp)
            
        x_temp = gated_multiscale_mixer(x_temp, filters=d_model, name_prefix=f"mixer_{i}")
        
        # Reshape back to (B, Time, Nodes, Channels)
        x_temp = tf.reshape(x_temp, [b_shape, n_shape, t_shape, d_model])
        x = tf.transpose(x_temp, [0, 2, 1, 3]) 
        
        # B. SPATIAL MIXING (Time-independent)
        # Einsum smoothly handles the 4D tensor: multiplies node features by Adjacency matrix
        x_spat = Lambda(
            lambda args: tf.einsum("ij,btjc->btic", args[0], args[1]), 
            name=f"spatial_gcn_{i}"
        )([A_matrix, x])
        x_spat = Dense(d_model, activation="swish", kernel_regularizer=reg, name=f"spatial_dense_{i}")(x_spat)
        
        # Residual + Norm
        x = LayerNormalization(name=f"st_ln_{i}")(x + x_spat)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    # 5. Output Projection
    out = Dense(1, kernel_regularizer=reg, name="final_proj")(x)
    out = tf.squeeze(out, axis=-1) # Drop channel dim -> (Batch, Time, Nodes)

    # Learn the residual based on the interpolated input
    final_output = Add(name="final_output")([values, out])

    return Model(inp, final_output, name="imputation_model")
