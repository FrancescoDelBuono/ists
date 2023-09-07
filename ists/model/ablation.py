import tensorflow as tf

from .embedding import TemporalEmbedding, SpatialEmbedding
from .encoder import EncoderLayer


class TransformerTemporal(tf.keras.layers.Layer):
    def __init__(
            self,
            *,
            feature_mask,
            kernel_size,
            d_model,
            num_heads,
            dff,
            fff,
            activation='relu',
            time_cnn=True,
            dropout_rate=0.1,
            null_max_size=None,
            time_max_sizes=None,
            **kwargs
    ):
        super().__init__()

        self.temporal_embedder = TemporalEmbedding(
            d_model=d_model,
            kernel_size=kernel_size,
            feature_mask=feature_mask,
            with_cnn=time_cnn,
            null_max_size=null_max_size,
            time_max_sizes=time_max_sizes,
        )

        self.encoder = EncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
            activation=activation,
        )

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(fff, activation='gelu')
        self.final_layer = tf.keras.layers.Dense(1)

    def call(self, inputs, **kwargs):
        x = inputs[0]
        x = self.temporal_embedder(x)
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.dense(x)
        pred = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        return pred


class TransformerExogenous(tf.keras.layers.Layer):
    def __init__(
            self,
            *,
            exg_feature_mask,
            kernel_size,
            d_model,
            num_heads,
            dff,
            fff,
            activation='relu',
            exg_cnn=True,
            dropout_rate=0.1,
            exg_time_max_sizes=None,
            **kwargs
    ):
        super().__init__()

        self.temporal_embedder = TemporalEmbedding(
            d_model=d_model,
            kernel_size=kernel_size,
            with_cnn=exg_cnn,
            feature_mask=exg_feature_mask,
            time_max_sizes=exg_time_max_sizes,
        )

        self.encoder = EncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
            activation=activation,
        )

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(fff, activation='gelu')
        self.final_layer = tf.keras.layers.Dense(1)

    def call(self, inputs, **kwargs):
        x = inputs[1]
        x = self.temporal_embedder(x)
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.dense(x)
        pred = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        return pred


class TransformerSpatial(tf.keras.layers.Layer):
    def __init__(
            self,
            *,
            feature_mask,
            spatial_size,
            kernel_size,
            d_model,
            num_heads,
            dff,
            fff,
            activation='relu',
            spt_cnn=True,
            dropout_rate=0.1,
            null_max_size=None,
            time_max_sizes=None,
            **kwargs
    ):
        super().__init__()

        self.spatial_embedder = SpatialEmbedding(
            d_model=d_model,
            kernel_size=kernel_size,
            spatial_size=spatial_size,
            feature_mask=feature_mask,
            with_cnn=spt_cnn,
            null_max_size=null_max_size,
            time_max_sizes=time_max_sizes,
        )

        self.encoder = EncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
            activation=activation,
        )

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(fff, activation='gelu')
        self.final_layer = tf.keras.layers.Dense(1)

    def call(self, inputs, **kwargs):
        x = inputs[2:]
        x = self.spatial_embedder(x)
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.dense(x)
        pred = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        return pred


class TransformerTemporalSpatial(tf.keras.layers.Layer):
    def __init__(
            self,
            *,
            feature_mask,
            exg_feature_mask,
            spatial_size,
            kernel_size,
            d_model,
            num_heads,
            dff,
            fff,
            activation='relu',
            exg_cnn=True,
            spt_cnn=True,
            time_cnn=True,
            dropout_rate=0.1,
            null_max_size=None,
            time_max_sizes=None,
            exg_time_max_sizes=None,
    ):
        super().__init__()

        self.temporal_embedder = TemporalEmbedding(
            d_model=d_model,
            kernel_size=kernel_size,
            feature_mask=feature_mask,
            with_cnn=time_cnn,
            null_max_size=null_max_size,
            time_max_sizes=time_max_sizes,
        )

        self.temporal_encoder = EncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
            activation=activation,
        )

        self.spatial_embedder = SpatialEmbedding(
            d_model=d_model,
            kernel_size=kernel_size,
            spatial_size=spatial_size,
            feature_mask=feature_mask,
            with_cnn=spt_cnn,
            null_max_size=null_max_size,
            time_max_sizes=time_max_sizes,
        )

        self.spatial_encoder = EncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
            activation=activation,
        )

        self.global_encoder = EncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
            activation=activation,
        )

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(fff, activation='gelu')
        self.final_layer = tf.keras.layers.Dense(1)

    def call(self, inputs, **kwargs):
        temporal_x = inputs[0]
        # exogenous_x = inputs[1]
        spatial_array = inputs[2:]

        # Temporal Embedding and Encoder
        temporal_x = self.temporal_embedder(temporal_x)
        temporal_x = self.temporal_encoder(temporal_x)

        # Spatial Embedding and Encoder
        spatial_x = self.spatial_embedder(spatial_array)
        spatial_x = self.spatial_encoder(spatial_x)

        # Global Encoder
        embedded_x = tf.concat([temporal_x, spatial_x], axis=1)
        embedded_x = self.global_encoder(embedded_x)

        embedded_x = self.flatten(embedded_x)
        embedded_x = self.dense(embedded_x)
        pred = self.final_layer(embedded_x)

        return pred


class TransformerSpatialExogenous(tf.keras.layers.Layer):
    def __init__(
            self,
            *,
            feature_mask,
            exg_feature_mask,
            spatial_size,
            kernel_size,
            d_model,
            num_heads,
            dff,
            fff,
            activation='relu',
            exg_cnn=True,
            spt_cnn=True,
            time_cnn=True,
            dropout_rate=0.1,
            null_max_size=None,
            time_max_sizes=None,
            exg_time_max_sizes=None,
    ):
        super().__init__()

        self.exogenous_embedder = TemporalEmbedding(
            d_model=d_model,
            kernel_size=kernel_size,
            feature_mask=exg_feature_mask,
            with_cnn=exg_cnn,
            time_max_sizes=exg_time_max_sizes,
        )

        self.exogenous_encoder = EncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
            activation=activation,
        )

        self.spatial_embedder = SpatialEmbedding(
            d_model=d_model,
            kernel_size=kernel_size,
            spatial_size=spatial_size,
            feature_mask=feature_mask,
            with_cnn=spt_cnn,
            null_max_size=null_max_size,
            time_max_sizes=time_max_sizes,
        )

        self.spatial_encoder = EncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
            activation=activation,
        )

        self.global_encoder = EncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
            activation=activation,
        )

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(fff, activation='gelu')
        self.final_layer = tf.keras.layers.Dense(1)

    def call(self, inputs, **kwargs):
        # temporal_x = inputs[0]
        exogenous_x = inputs[1]
        spatial_array = inputs[2:]

        # Temporal Embedding and Encoder
        exogenous_x = self.exogenous_embedder(exogenous_x)
        exogenous_x = self.exogenous_embedder(exogenous_x)

        # Spatial Embedding and Encoder
        spatial_x = self.spatial_embedder(spatial_array)
        spatial_x = self.spatial_encoder(spatial_x)

        # Global Encoder
        embedded_x = tf.concat([exogenous_x, spatial_x], axis=1)
        embedded_x = self.global_encoder(embedded_x)

        embedded_x = self.flatten(embedded_x)
        embedded_x = self.dense(embedded_x)
        pred = self.final_layer(embedded_x)

        return pred


class TransformerTemporalExogenous(tf.keras.layers.Layer):

    def __init__(
            self,
            *,
            feature_mask,
            exg_feature_mask,
            spatial_size,
            kernel_size,
            d_model,
            num_heads,
            dff,
            fff,
            activation='relu',
            exg_cnn=True,
            spt_cnn=True,
            time_cnn=True,
            dropout_rate=0.1,
            null_max_size=None,
            time_max_sizes=None,
            exg_time_max_sizes=None,
    ):
        super().__init__()

        self.temporal_embedder = TemporalEmbedding(
            d_model=d_model,
            kernel_size=kernel_size,
            feature_mask=feature_mask,
            with_cnn=time_cnn,
            null_max_size=null_max_size,
            time_max_sizes=time_max_sizes,
        )

        self.temporal_encoder = EncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
            activation=activation,
        )

        self.exogenous_embedder = TemporalEmbedding(
            d_model=d_model,
            kernel_size=kernel_size,
            with_cnn=exg_cnn,
            feature_mask=exg_feature_mask,
            time_max_sizes=exg_time_max_sizes,
        )

        self.exogenous_encoder = EncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
            activation=activation,
        )

        self.global_encoder = EncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
            activation=activation,
        )

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(fff, activation='gelu')
        self.final_layer = tf.keras.layers.Dense(1)

    def call(self, inputs, **kwargs):
        temporal_x = inputs[0]
        exogenous_x = inputs[1]
        # spatial_array = inputs[2:]

        # Temporal Embedding and Encoder
        temporal_x = self.temporal_embedder(temporal_x)
        temporal_x = self.temporal_encoder(temporal_x)

        # Spatial Embedding and Encoder
        exogenous_x = self.spatial_embedder(exogenous_x)
        exogenous_x = self.spatial_encoder(exogenous_x)

        # Global Encoder
        embedded_x = tf.concat([temporal_x, exogenous_x], axis=1)
        embedded_x = self.global_encoder(embedded_x)

        embedded_x = self.flatten(embedded_x)
        embedded_x = self.dense(embedded_x)
        pred = self.final_layer(embedded_x)

        return pred
