import tensorflow as tf

from .embedding import TemporalEmbedding, SpatialEmbedding
from .encoder import EncoderLayer, CrossEncoderLayer, GlobalEncoderLayer


class TransformerTemporal(tf.keras.Model):
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


class TransformerExogenous(tf.keras.Model):
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


class TransformerSpatial(tf.keras.Model):
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


class TransformerTemporalSpatial(tf.keras.Model):
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


class TransformerSpatialExogenous(tf.keras.Model):
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
            feature_mask=exg_feature_mask,
            with_cnn=exg_cnn,
            time_max_sizes=exg_time_max_sizes,
        )

        self.exogenous_encoder = CrossEncoderLayer(
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
        exogenous_x = inputs[1]
        spatial_array = inputs[2:]

        # Temporal Embedding and Encoder
        temporal_x = self.temporal_embedder(temporal_x)
        temporal_x = self.temporal_encoder(temporal_x)

        # Exogenous Embedding and Encoder
        exogenous_x = self.exogenous_embedder(exogenous_x)
        exogenous_x = self.exogenous_encoder(x=exogenous_x, context=temporal_x)

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


class TransformerTemporalExogenous(tf.keras.Model):

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

        self.exogenous_encoder = CrossEncoderLayer(
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
        exogenous_x = self.exogenous_embedder(exogenous_x)
        exogenous_x = self.exogenous_encoder(x=exogenous_x, context=temporal_x)

        # Global Encoder
        embedded_x = tf.concat([temporal_x, exogenous_x], axis=1)
        embedded_x = self.global_encoder(embedded_x)

        embedded_x = self.flatten(embedded_x)
        embedded_x = self.dense(embedded_x)
        pred = self.final_layer(embedded_x)

        return pred


class STTnoEmbedding(tf.keras.Model):
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
            num_layers=1,
            with_cross=True,
            dropout_rate=0.1,
            null_max_size=None,
            time_max_sizes=None,
            exg_time_max_sizes=None,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.with_cross = with_cross
        if with_cross:
            exo_encoder_layer = CrossEncoderLayer
        else:
            exo_encoder_layer = EncoderLayer

        self.temporal_encoders = [
            EncoderLayer(
                d_model=len(feature_mask),
                num_heads=num_heads,
                dff=dff,
                dropout_rate=dropout_rate,
                activation=activation,
            )
            for _ in range(self.num_layers)
        ]
        self.spatial_encoders = [
            EncoderLayer(
                d_model=len(feature_mask),
                num_heads=num_heads,
                dff=dff,
                dropout_rate=dropout_rate,
                activation=activation,
            )
            for _ in range(self.num_layers)
        ]

        self.exogenous_encoders = [
            exo_encoder_layer(
                d_model=len(exg_feature_mask),
                num_heads=num_heads,
                dff=dff,
                dropout_rate=dropout_rate,
                activation=activation,
            )
            for _ in range(self.num_layers)
        ]

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(fff, activation='gelu')
        self.final_layer = tf.keras.layers.Dense(1)
        self.last_attn_scores = None

    def call(self, inputs, **kwargs):
        time_x = inputs[0]
        exogenous_x = inputs[1]
        spatial_array = inputs[2:]
        spatial_x = tf.concat(spatial_array, axis=1)

        # temporal_emb, exogenous_emb, spatial_emb = self.encoder(temporal_x, exogenous_x, spatial_x)
        for i in range(self.num_layers):
            # Compute temporal, spatial, and exogenous embedding
            time_x = self.temporal_encoders[i](x=time_x)
            spatial_x = self.spatial_encoders[i](x=spatial_x)
            if self.with_cross:
                exogenous_x = self.exogenous_encoders[i](x=exogenous_x, context=time_x)
            else:
                exogenous_x = self.exogenous_encoders[i](x=exogenous_x)

        embedded_x = tf.concat([self.flatten(time_x), self.flatten(spatial_x), self.flatten(exogenous_x)], axis=1)
        embedded_x = self.dense(embedded_x)
        pred = self.final_layer(embedded_x)
        return pred


class TSWithExogenousFeatures(tf.keras.Model):
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
            **kwargs
    ):
        super().__init__()

        self.temporal_embedder = TemporalEmbedding(
            d_model=d_model,
            kernel_size=kernel_size,
            feature_mask=feature_mask + [x for x in exg_feature_mask if x == 0],
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

        self.exg_ids = [i for i, x in enumerate(exg_feature_mask) if x == 0]

    def call(self, inputs, **kwargs):
        temporal_x = inputs[0]
        exogenous_x = inputs[1]
        spatial_array = inputs[2:]

        exg_feat = tf.gather(exogenous_x, self.exg_ids, axis=2)

        temporal_x = tf.concat([temporal_x, exg_feat], axis=2)

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


class STTWithSpatialExogenous(tf.keras.Model):
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
            num_layers=1,
            with_cross=True,
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
        self.temporal_embedder_exg = TemporalEmbedding(
            d_model=d_model,
            kernel_size=kernel_size,
            feature_mask=feature_mask,
            with_cnn=time_cnn,
            null_max_size=null_max_size,
            time_max_sizes=time_max_sizes,
        )

        self.exogenous_embedder = TemporalEmbedding(
            d_model=d_model,
            kernel_size=kernel_size,
            with_cnn=exg_cnn,
            feature_mask=exg_feature_mask,
            time_max_sizes=exg_time_max_sizes,
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

        self.encoder = GlobalEncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            activation=activation,
            num_layers=num_layers,
            with_cross=with_cross,
            dropout_rate=dropout_rate
        )

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(fff, activation='gelu')
        self.final_layer = tf.keras.layers.Dense(1)
        self.last_attn_scores = None

    def call(self, inputs, **kwargs):
        temporal_x = inputs[0]
        exogenous_x = inputs[1]
        spatial_array = inputs[2:]

        temporal_x = self.temporal_embedder(temporal_x)
        # temporal_exg_x = self.temporal_embedder_exg(t_x)
        exogenous_x = self.exogenous_embedder(exogenous_x)
        spatial_x = self.spatial_embedder(spatial_array)

        exogenous_x = tf.concat([exogenous_x, temporal_x], axis=1)

        temporal_emb, exogenous_emb, spatial_emb = self.encoder(temporal_x, exogenous_x, spatial_x)
        embedded_x = tf.concat([temporal_emb, exogenous_emb, spatial_emb], axis=1)
        embedded_x = self.flatten(embedded_x)
        embedded_x = self.dense(embedded_x)
        pred = self.final_layer(embedded_x)

        return pred


class SEWithSpatialExogenous(tf.keras.Model):
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

        # self.temporal_encoder = EncoderLayer(
        #     d_model=d_model,
        #     num_heads=num_heads,
        #     dff=dff,
        #     dropout_rate=dropout_rate,
        #     activation=activation,
        # )

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
        temporal_x = inputs[0]
        exogenous_x = inputs[1]
        spatial_array = inputs[2:]

        # Temporal Embedding and Encoder
        temporal_x = self.temporal_embedder(temporal_x)
        # temporal_x = self.temporal_encoder(temporal_x)

        # Exogenous Embedding and Encoder
        exogenous_x = self.exogenous_embedder(exogenous_x)
        exogenous_x = tf.concat([exogenous_x, temporal_x], axis=1)

        exogenous_x = self.exogenous_encoder(x=exogenous_x)

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

