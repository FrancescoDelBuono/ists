from abc import ABC
from typing import TypeVar, List

import os
import numpy as np
import tensorflow as tf

from .model import STTransformer, BaselineModel
from ..preprocessing import StandardScalerBatch, MinMaxScalerBatch
from ..metrics import compute_metrics
from .ablation import TransformerSpatial, TransformerTemporal, TransformerExogenous, TransformerTemporalSpatial, \
    TransformerSpatialExogenous, TransformerTemporalExogenous, STTnoEmbedding
from .ablation import TSWithExogenousFeatures, STTWithSpatialExogenous, SEWithSpatialExogenous

T = TypeVar('T', bound=tf.keras.Model)


def get_transformer(transform_type: str) -> object:
    # Return the selected model
    if transform_type == 'standard':
        return StandardScalerBatch(p1=None, p2=None)
    elif transform_type == 'standard01':
        return StandardScalerBatch(p1=1, p2=99)
    elif transform_type == 'minmax':
        return MinMaxScalerBatch()
    else:
        raise ValueError('Transformer {} is not supported, it must be "standard" or "minmax"')


def get_model(model_type: str, model_params) -> T:
    # Return the selected model
    if model_type == 'sttransformer':
        return STTransformer(**model_params)
    elif model_type == 'stt_no_embd':
        return STTnoEmbedding(**model_params)
    elif model_type == 'dense':
        return BaselineModel(feature_mask=model_params['feature_mask'], base_model='dense',
                             hidden_units=model_params['d_model'], skip_na=False, activation='gelu')
    elif model_type == 'lstm':
        return BaselineModel(feature_mask=model_params['feature_mask'], base_model='lstm',
                             hidden_units=model_params['d_model'], skip_na=True, activation='gelu')
    elif model_type == 'bilstm':
        return BaselineModel(feature_mask=model_params['feature_mask'], base_model='bilstm',
                             hidden_units=model_params['d_model'], skip_na=True, activation='gelu')
    elif model_type == 'lstm_base':
        return BaselineModel(feature_mask=model_params['feature_mask'], base_model='lstm',
                             hidden_units=model_params['d_model'], skip_na=False, activation='gelu')
    elif model_type == 'bilstm_base':
        return BaselineModel(feature_mask=model_params['feature_mask'], base_model='bilstm',
                             hidden_units=model_params['d_model'], skip_na=False, activation='gelu')
    elif model_type == 't':
        return TransformerTemporal(**model_params)
    elif model_type == 's':
        return TransformerSpatial(**model_params)
    elif model_type == 'e':
        return TransformerExogenous(**model_params)
    elif model_type == 'ts':
        return TransformerTemporalSpatial(**model_params)
    elif model_type == 'te':
        return TransformerTemporalExogenous(**model_params)
    elif model_type == 'se':
        return TransformerSpatialExogenous(**model_params)
    elif model_type == 'ts_fe':
        return TSWithExogenousFeatures(**model_params)
    elif model_type == 'stt_se':
        return STTWithSpatialExogenous(**model_params)
    elif model_type == 'se_se':
        return SEWithSpatialExogenous(**model_params)
    else:
        raise ValueError('Model {} is not supported, it must be "sttransformer"')


def custom_mae_loss(y_true, y_pred):
    factor_levels = tf.unique(y_true[:, 1]).y
    loss = tf.constant(0.0)

    for level in factor_levels:
        mask = tf.equal(y_true[:, 1], level)
        true_subset = tf.boolean_mask(y_true[:, 0], mask)
        pred_subset = tf.boolean_mask(y_pred[:, 0], mask)
        mae = tf.reduce_mean(tf.abs(true_subset - pred_subset))
        loss += (1.0 / tf.cast(level, dtype=tf.float32)) * mae

    return loss


def custom_mse_loss(y_true, y_pred):
    factor_levels = tf.unique(y_true[:, 1]).y
    loss = tf.constant(0.0)

    for level in factor_levels:
        mask = tf.equal(y_true[:, 1], level)
        true_subset = tf.boolean_mask(y_true[:, 0], mask)
        pred_subset = tf.boolean_mask(y_pred[:, 0], mask)
        mse = tf.reduce_mean(tf.square(true_subset - pred_subset))
        loss += (1.0 / tf.cast(level, dtype=tf.float32)) * mse

    return loss


class FunctionCallback(tf.keras.callbacks.Callback):
    def __init__(self, x: np.ndarray, spt: np.ndarray, exg: np.ndarray, y: np.ndarray, transformer: object = None):
        super(FunctionCallback, self).__init__()
        self.x = x
        self.spt = spt
        self.exg = exg
        self.y = y
        self.transformer = transformer

    def _label_inverse_transform(self, y):
        if self.transformer is not None:
            y = np.copy(y)
            y = self.transformer.inverse_transform(y)
        return y

    def on_epoch_end(self, epoch, logs=None):
        # Get predictions for the subset of data
        y_pred_subset = self.model.predict([self.x] + [self.exg] + self.spt, verbose=0)

        # Compute metrics on the subset on the transformed data domain
        metrics = compute_metrics(self.y, y_pred_subset)
        metrics = " ".join([f'{k}:{val:.4f}' for k, val in metrics.items()])
        print("Metrics trf epoch {}: {}".format(epoch, metrics))

        # Compute metrics on the subset on the raw data domain
        y_true_raw = self._label_inverse_transform(self.y)
        y_pred_raw = self._label_inverse_transform(y_pred_subset)
        metrics = compute_metrics(y_true_raw, y_pred_raw)
        metrics = " ".join([f'{k}:{val:.4f}' for k, val in metrics.items()])
        print("Metrics raw epoch {}: {}".format(epoch, metrics))


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule, ABC):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps
        }


def delete_non_empty_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            delete_non_empty_directory(file_path)  # Recursively delete subdirectories
    os.rmdir(directory_path)


class ModelWrapper(object):
    def __init__(
            self,
            checkpoint_dir: str,
            model_type: str,
            model_params: dict,
            transform_type: str = None,
            loss: str = 'mse',
            lr: float = 0.001,
            best_valid: bool = True
    ):
        self.model = get_model(model_type, model_params)
        self.transform_type = transform_type
        if transform_type:
            self.transformer = get_transformer(transform_type)
            self.spt_transformer = get_transformer(transform_type)
            self.exg_transformer = get_transformer(transform_type)

        self.checkpoint_delete_folder = False
        self.checkpoint_basedir = checkpoint_dir
        self.checkpoint_dir = os.path.join(self.checkpoint_basedir, 'best_model')
        self.checkpoint_path = os.path.join(self.checkpoint_basedir, 'best_model', 'cp.ckpt')

        self.best_valid = best_valid
        self.loss = loss
        self.lr = lr

        self.history = None

        self.d_model = model_params['d_model']
        self.feature_mask = model_params['feature_mask']
        self.exg_feature_mask = model_params['exg_feature_mask']

        # Check if model output dir exists
        if not os.path.isdir(self.checkpoint_basedir):
            os.makedirs(self.checkpoint_basedir, exist_ok=True)
            self.checkpoint_delete_folder = True
            # raise ValueError(f'Model output dir does not exist: {checkpoint_dir}')

        # Create checkpoint directory if it does not exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _fit_transform(self, x: np.ndarray, spt: List[np.ndarray], exg: np.ndarray):
        if self.transform_type:
            x = np.copy(x)
            spt = [np.copy(arr) for arr in spt]
            exg = np.copy(exg)

            cond_x = np.array(self.feature_mask) == 0
            x[:, :, cond_x] = self.transformer.fit_transform(x[:, :, cond_x])

            spt_size = len(spt)
            spt_all = np.concatenate(spt, axis=1)
            spt_all[:, :, cond_x] = self.spt_transformer.fit_transform(spt_all[:, :, cond_x])
            spt = np.split(spt_all, spt_size, axis=1)

            cond_exg = np.array(self.exg_feature_mask) == 0
            exg[:, :, cond_exg] = self.exg_transformer.fit_transform(exg[:, :, cond_exg])
        # elif self.transform_type.startswith('standard') or self.transform_type.startswith('standard'):
        #     x = np.copy(x)
        #     spt = [np.copy(arr) for arr in spt]
        #
        #     idx_x = self.transform_type.split('_')[1:]
        #     idx_x = [int(idx) for idx in idx_x]
        #
        #     x[:, :, idx_x] = self.transformer.fit_transform(x[:, :, idx_x])
        #
        #     spt_size = len(spt)
        #     spt_all = np.concatenate(spt, axis=1)
        #     spt_all[:, :, idx_x] = self.spt_transformer.fit_transform(spt_all[:, :, idx_x])
        #     spt = np.split(spt_all, spt_size, axis=1)

        return x, spt, exg

    def _label_transform(self, y):
        if self.transform_type:
            y = np.copy(y)
            y = self.transformer.transform(y)

        return y

    def _label_inverse_transform(self, y):
        y = np.copy(y)
        if self.transform_type:
            y = self.transformer.inverse_transform(y)
        return y

    def _remove_model_checkpoint(self):
        if os.path.isdir(self.checkpoint_dir):
            delete_non_empty_directory(self.checkpoint_dir)
        if self.checkpoint_delete_folder:
            os.rmdir(self.checkpoint_basedir)

    def _get_best_model(self):
        if self.best_valid and not os.path.isdir(self.checkpoint_dir):
            raise ValueError('Impossible to load saved model, it does not exist!')

        if self.best_valid:
            # self.model = tf.keras.models.load_model(self.model_path)
            self.model.load_weights(self.checkpoint_path)

    @staticmethod
    def _get_spatial_array(x: np.ndarray, spt: List[np.ndarray]) -> List[np.ndarray]:
        spt_num_past = spt[0].shape[1]
        spt_x = [np.copy(x[:, -spt_num_past:, :])] + spt
        return spt_x

    def fit(
            self,
            x: np.ndarray,
            spt: List[np.ndarray],
            exg: np.ndarray,
            y: np.ndarray,
            epochs: int = 50,
            batch_size: int = 32,
            validation_split: float = 0.1,
            verbose: int = 0,
            extra=None,
    ):
        spt = self._get_spatial_array(x, spt)
        x, spt, exg = self._fit_transform(x, spt, exg)
        y = self._label_transform(y)

        # x_test, spt_test, exg_test, y_test = extra['x'], extra['spt'], extra['exg'], extra['y']
        # x_test, spt_test, exg_test = self._fit_transform(x_test, spt_test, exg_test)
        # y_test = self._label_transform(y_test)
        #
        # test_callback = FunctionCallback(
        #     x_test,
        #     spt_test,
        #     exg_test,
        #     y_test,
        #     transformer=self.transformer if self.transform_type else None
        # )
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            self.checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            mode='min',
            # verbose=1,
            verbose=0
            # save_format='tf'
        )
        if self.lr > 0:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)  # , clipnorm=1.0, clipvalue=0.5)
        else:
            learning_rate = CustomSchedule(self.d_model)
            optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        self.model.compile(
            loss=self.loss,
            optimizer=optimizer,
            metrics=['mae', 'mse'],
            # run_eagerly=True,
        )

        self.history = self.model.fit(
            x=[x] + [exg] + spt,
            y=y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            # verbose=verbose,
            verbose=0,
            callbacks=[model_checkpoint]
        )

        # Load best model
        self._get_best_model()
        self._remove_model_checkpoint()

    def predict(self, x: np.ndarray, spt: List[np.ndarray], exg: np.ndarray):
        spt = self._get_spatial_array(x, spt)
        x, spt, exg = self._fit_transform(x, spt, exg)

        y_preds = self.model.predict([x] + [exg] + spt, verbose=0)

        y_preds = self._label_inverse_transform(y_preds)

        return y_preds

# # Plotting the validation and training loss
# history = model.history
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
