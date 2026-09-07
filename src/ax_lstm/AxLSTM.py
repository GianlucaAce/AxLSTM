"""
AxLSTM.py
=========
Standalone module for building, training, and evaluating LSTM networks on univariate time series.

It incorporates functionalities previously found in AxLib.py as private methods.
It uses Keras standalone.

Typical usage
----------
    config = {
        'time_steps': 7,
        'split_value': 5,
        'batch_size': 32,
        'units': [64, 32],
        'dropout': 0.2,
        'learning_rate': 0.001,
        'stateful': False,
        'early_stopping_patience': 10,
    }
    model = AxLSTM(config)
    model.load_and_prepare('data/Test.txt', delimiter='\\t')
    model.build_model(verbose=True)
    model.train(epochs=100)
    model.evaluate()
    model.plot_history()
    model.plot_predictions()
"""

import os
import sys
import time
from typing import Dict, Any, List, Tuple, Optional, Union

# Suppressing Informational Messages and TensorFlow's oneDNN C++
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, History


class _StatefulEarlyStopping(EarlyStopping):
    """Preserve EarlyStopping state across one-epoch fit calls."""

    def on_train_begin(self, logs=None):
        """Initialize EarlyStopping only once during stateful training."""
        if getattr(self, '_stateful_started', False):
            return
        self._stateful_started = True
        super().on_train_begin(logs)

class AxLSTM:
    """Toolbox for LSTM networks on univariate time series.

    Parameters
    ----------
    config : dict
        Dictionary of configuration and hyperparameters. Supported keys:
        - ``time_steps``               (int,   default 7)    — sliding window size
        - ``split_value``              (int,   default 5)    — train/val split ratio (1/N of total in val)
        - ``batch_size``               (int,   default 32)   — mini-batch size
        - ``units``                    (list,  default [64]) — list of units/neurons for each LSTM layer
        - ``dropout``                  (float, default 0.2)  — dropout rate between LSTM layers
        - ``learning_rate``            (float, default 0.001)— learning rate for Adam optimizer
        - ``stateful``                 (bool,  default False)— LSTM stateful mode
        - ``early_stopping_patience``  (int,   default 0)    — EarlyStopping patience (0 = disabled)
        - ``norm_min``                 (float, default 0.0)  — lower bound for MinMaxScaler
        - ``norm_max``                 (float, default 1.0)  — upper bound for MinMaxScaler
    """

    _DEFAULTS: Dict[str, Any] = {
        'time_steps': 7,
        'split_value': 5,
        'batch_size': 32,
        'units': [64],
        'dropout': 0.2,
        'learning_rate': 0.001,
        'stateful': False,
        'early_stopping_patience': 0,
        'norm_min': 0.0,
        'norm_max': 1.0,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the LSTM pipeline with optional configuration overrides.

        Parameters
        ----------
        config : dict, optional
            Configuration values overriding the class defaults.
        """
        user_config = config or {}
        self.config: Dict[str, Any] = {**self._DEFAULTS, **user_config}

        # Scaler and model
        self.scaler: Optional[MinMaxScaler] = None
        self.model: Optional[Sequential] = None
        self.history: Any = None

        # Tensors prepared for LSTM: (samples, time_steps, features)
        self.t_x: Optional[np.ndarray] = None
        self.t_y: Optional[np.ndarray] = None
        self.v_x: Optional[np.ndarray] = None
        self.v_y: Optional[np.ndarray] = None

        # Internal timer
        self._timer_start: Optional[float] = None

    # =======================================================================
    # PUBLIC API - PREPARATION, BUILDING, TRAINING, EVALUATION
    # =======================================================================

    def load_and_prepare(
        self,
        filepath: str,
        delimiter: str = '\t',
        target_col: str = 'y',
        date_col: str = 'data',
        drop_date: bool = True,
        clean_mode: str = 'drop',
        clean_lower: Optional[float] = None,
        clean_upper: Optional[float] = None,
        verbose: bool = False,
    ):
        """Load data file (CSV/TSV), performs cleaning, normalization, featurization,
        split and batch size adjustment compatible for LSTM.

        Parameters
        ----------
        filepath : str
            Path to the data file.
        delimiter : str
            Column delimiter in the file (default '\\t').
        target_col : str
            Target column name to predict (default 'y').
        date_col : str
            Name of the date column (default 'data').
        drop_date : bool
            Whether to drop the date column if present.
        clean_mode : str
            'drop' to remove rows outside thresholds, 'clip' to confine them to thresholds, 'none' for no cleaning.
        clean_lower : float, optional
            Lower bound for data cleaning.
        clean_upper : float, optional
            Upper bound for data cleaning.
        verbose : bool
            If True, prints statistics of the various steps.
        """
        cfg = self.config

        # 1. Load data
        data = self._load_data(filepath, delimiter, date_col, drop_date, verbose)

        # 2. Data cleaning
        if clean_mode != 'none':
            data = self._clean_data(
                data,
                target_col=target_col,
                lower=clean_lower,
                upper=clean_upper,
                mode=clean_mode,
                verbose=verbose
            )

        # 3. Normalization
        self.scaler, data_scaled = self._normalize(
            data[[target_col]],
            norm_min=cfg['norm_min'],
            norm_max=cfg['norm_max'],
            verbose=verbose
        )

        # 4. Split Train / Validation (chronological without shuffle)
        data_train, data_val = self._split_data(
            data_scaled,
            split_value=cfg['split_value'],
            verbose=verbose
        )

        # 5. Featurize (sliding window)
        data_train_f, col_names = self._featurize(
            data_train,
            n_shift=cfg['time_steps'],
            inverse=True
        )
        data_val_f, _ = self._featurize(
            data_val,
            n_shift=cfg['time_steps'],
            inverse=True
        )

        if verbose:
            print(f"[load_and_prepare] Featurized train: {len(data_train_f)}, "
                  f"val: {len(data_val_f)}, cols: {col_names}")

        # 6. Separation Features and Target Y
        t_x, t_y = self._split_xy(data_train_f, col_names, label_col='Y')
        v_x, v_y = self._split_xy(data_val_f, col_names, label_col='Y')

        # 7. Adjustment of the tensor length to the batch_size
        t_x, t_y, v_x, v_y = self._fit_batch_size(
            cfg['batch_size'], t_x, t_y, v_x, v_y, verbose=verbose
        )

        # 8. Reshape for 3D tensor (samples, time_steps, features=1)
        self.t_x = t_x.reshape(t_x.shape[0], t_x.shape[1], 1)
        self.v_x = v_x.reshape(v_x.shape[0], v_x.shape[1], 1)
        self.t_y = t_y
        self.v_y = v_y

        if verbose:
            print(f"[load_and_prepare] Tensors formed:")
            print(f"  t_x: {self.t_x.shape}, t_y: {self.t_y.shape}")
            print(f"  v_x: {self.v_x.shape}, v_y: {self.v_y.shape}")

    def build_model(self, verbose: bool = False) -> Sequential:
        """Builds and compiles the Keras model with unidirection LSTM.

        Parameters
        ----------
        verbose : bool
            If True, prints model.summary().

        Returns
        -------
        Sequential
            Compiled model.
        """
        if self.t_x is None:
            raise RuntimeError("Call load_and_prepare() before build_model().")

        cfg = self.config
        units_list = cfg['units']
        if isinstance(units_list, int):
            units_list = [units_list]
        dropout_rate = cfg['dropout']
        stateful = cfg['stateful']
        batch_size = cfg['batch_size']
        time_steps = cfg['time_steps']

        model = Sequential(name="AxLSTM_Model")

        # Explicit input layer (Keras 3)
        if stateful:
            model.add(Input(batch_shape=(batch_size, time_steps, 1)))
        else:
            model.add(Input(shape=(time_steps, 1)))

        for idx, units in enumerate(units_list):
            is_last = (idx == len(units_list) - 1)
            return_sequences = not is_last

            lstm_kwargs: Dict[str, Any] = {
                'units': units,
                'return_sequences': return_sequences,
                'stateful': stateful,
                'name': f"lstm_layer_{idx}",
            }

            model.add(LSTM(**lstm_kwargs))

            if not is_last and dropout_rate > 0.0:
                model.add(Dropout(rate=dropout_rate, name=f"dropout_layer_{idx}"))

        # Output layer for scalar regression
        model.add(Dense(1, name="output_dense"))

        optimizer = Adam(learning_rate=cfg['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse')

        self.model = model

        if verbose:
            model.summary()

        return model

    def train(
        self,
        epochs: int = 100,
        shuffle: bool = True,
        verbose: int = 1,
    ):
        """Train the Keras model.

        Parameters
        ----------
        epochs : int
            Number of training epochs.
        shuffle : bool
            If shuffle samples at each epoch (ignored if stateful=True).
        verbose : int
            Keras verbosity level.

        Returns
        -------
        keras.callbacks.History
            The History object containing the training metrics.
        """
        if self.model is None:
            raise RuntimeError("Call build_model() before train().")

        cfg = self.config
        callbacks = []
        patience = cfg.get('early_stopping_patience', 0)
        if patience > 0:
            stopping_callback = _StatefulEarlyStopping if cfg['stateful'] else EarlyStopping
            callbacks.append(
                stopping_callback(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1
                )
            )

        start = self._start_timer()

        if cfg['stateful']:
            # Stateful training: manual reset of states at each epoch
            combined_history = History()
            combined_history.history = {}
            self.model.stop_training = False
            for ep in range(epochs):
                epoch_history = self.model.fit(
                    self.t_x, self.t_y,
                    epochs=1,
                    batch_size=cfg['batch_size'],
                    validation_data=(self.v_x, self.v_y),
                    shuffle=False,
                    callbacks=callbacks,
                    verbose=verbose if verbose > 1 else 0
                )
                for metric, values in epoch_history.history.items():
                    combined_history.history.setdefault(metric, []).extend(values)
                for layer in self.model.layers:
                    if hasattr(layer, 'reset_states'):
                        layer.reset_states()
                if verbose == 1:
                    self._update_progress((ep + 1) / epochs)
                if getattr(self.model, 'stop_training', False):
                    break
            if verbose == 1:
                print()
            self.history = combined_history
        else:
            self.history = self.model.fit(
                self.t_x, self.t_y,
                epochs=epochs,
                batch_size=cfg['batch_size'],
                validation_data=(self.v_x, self.v_y),
                shuffle=shuffle,
                callbacks=callbacks,
                verbose=verbose
            )

        elapsed = time.time() - start
        self._time_elapsed(elapsed)
        print("[Done]")
        return self.history

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the loss on training set and validation set.

        Returns
        -------
        dict
            Dictionary containing 'train_loss' and 'val_loss'.
        """
        if self.model is None or self.t_x is None or self.v_x is None:
            raise RuntimeError("Model not trained or data not present.")

        cfg = self.config
        train_loss = float(self.model.evaluate(self.t_x, self.t_y, batch_size=cfg['batch_size'], verbose=0))
        val_loss = float(self.model.evaluate(self.v_x, self.v_y, batch_size=cfg['batch_size'], verbose=0))

        print(f"[evaluate] train_loss (MSE): {train_loss:.6f} | val_loss (MSE): {val_loss:.6f}")
        return {'train_loss': train_loss, 'val_loss': val_loss}

    def predict(self, data: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate predictions on a tensor or on the validation set.

        Parameters
        ----------
        data : np.ndarray, optional
            Input tensor (samples, time_steps, 1). If None, uses self.v_x.

        Returns
        -------
        np.ndarray
            Array of predictions in normalized scale.
        """
        if self.model is None:
            raise RuntimeError("The model is not yet ready or trained.")

        input_data = self.v_x if data is None else data
        return self.model.predict(input_data, batch_size=self.config['batch_size'])

    # =======================================================================
    # PLOTTING AND VISUALIZATION
    # =======================================================================

    def plot_history(self, title: str = 'Curva di Loss (Denormalizzata)', show: bool = True):
        """Plots the training loss (train vs validation) denormalized.
        Calculates sqrt(loss) and applies denormalization via the scaler.
        """
        if self.history is None or 'loss' not in self.history.history:
            raise RuntimeError("No history available to plot.")
        if self.scaler is None:
            raise RuntimeError("Scaler not initialized.")

        train_loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        ein_sqrt = [np.sqrt(v) for v in train_loss]
        eval_sqrt = [np.sqrt(v) for v in val_loss]

        ein_orig = self.scaler.inverse_transform(pd.DataFrame(ein_sqrt)).flatten().tolist()
        eval_orig = self.scaler.inverse_transform(pd.DataFrame(eval_sqrt)).flatten().tolist()

        plt.figure(figsize=(15, 4))
        plt.plot(ein_orig, label='Train')
        plt.plot(eval_orig, label='Validation')
        plt.title(title)
        plt.ylabel('Loss (original scale)')
        plt.xlabel('Epoch')
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.tight_layout()
        if show:
            plt.show()

    def plot_predictions(
        self,
        title: str = 'Prediction vs Actual Values',
        y_label: str = 'Value',
        x_label: str = 'Samples',
        show: bool = True
    ):
        """Compares predictions with actual values of the validation set."""
        if self.model is None or self.scaler is None or self.v_x is None or self.v_y is None:
            raise RuntimeError("Model or validation data not available.")

        preds = self.predict(self.v_x)
        preds_orig = self.scaler.inverse_transform(preds).flatten().tolist()
        real_orig = self.scaler.inverse_transform(self.v_y).flatten().tolist()

        plt.figure(figsize=(15, 4))
        plt.plot(preds_orig, label='Prediction')
        plt.plot(real_orig, label='Actual Values')
        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.grid(True)
        plt.legend(loc='upper left')
        plt.tight_layout()
        if show:
            plt.show()

    def plot_dashboard(
        self,
        title_history: str = 'Training Loss (Denormalized)',
        title_predictions: str = 'Prediction vs Actual Values',
        y_label: str = 'Value',
        x_label: str = 'Samples',
        show: bool = True
    ):
        """Shows both graphs in a single window with two panels (dashboard)."""
        if self.history is None or self.model is None or self.scaler is None or self.v_x is None or self.v_y is None:
            raise RuntimeError("Model or validation data not available to generate the dashboard.")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

        # 1. Loss Curve
        train_loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        ein_orig = self.scaler.inverse_transform(pd.DataFrame([np.sqrt(v) for v in train_loss])).flatten().tolist()
        eval_orig = self.scaler.inverse_transform(pd.DataFrame([np.sqrt(v) for v in val_loss])).flatten().tolist()

        ax1.plot(ein_orig, label='Train')
        ax1.plot(eval_orig, label='Validation')
        ax1.set_title(title_history)
        ax1.set_ylabel('Loss (original scale)')
        ax1.set_xlabel('Epoch')
        ax1.grid(True)
        ax1.legend(loc='upper right')

        # 2. Predictions vs Actual Values
        preds = self.predict(self.v_x)
        preds_orig = self.scaler.inverse_transform(preds).flatten().tolist()
        real_orig = self.scaler.inverse_transform(self.v_y).flatten().tolist()

        ax2.plot(preds_orig, label='Prediction')
        ax2.plot(real_orig, label='Actual Values')
        ax2.set_title(title_predictions)
        ax2.set_ylabel(y_label)
        ax2.set_xlabel(x_label)
        ax2.grid(True)
        ax2.legend(loc='upper left')

        plt.tight_layout()
        if show:
            plt.show()

    def plot_series(self, data: Any, title: str = 'Time Series'):
        """Plot of a single time series."""
        plt.figure(figsize=(15, 4))
        plt.plot(data)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_two(
        self,
        d1: Any, label1: str,
        d2: Any, label2: str,
        title: str = '',
        x_label: str = 'x',
        y_label: str = 'y',
        first_linewidth: float = 1.0
    ):
        """Plot of two series on the same graph."""
        plt.figure(figsize=(15, 4))
        plt.plot(d1, label=label1, linewidth=first_linewidth)
        plt.plot(d2, label=label2)
        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.grid(True)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

    def plot_min_max_med(
        self,
        hist_list: List[List[float]],
        length: int,
        plot_min: bool = True,
        plot_max: bool = True,
        plot_med: bool = True,
        show: bool = True
    ):
        """Plot Min, Max and Mean over N iterations. Fixes the hardcoded min_h bug."""
        if self.scaler is None:
            raise RuntimeError("Scaler not initialized.")

        n_cicli = len(hist_list)
        med_h = [0.0] * length
        min_h = [float('inf')] * length  
        max_h = [0.0] * length

        for i in range(length):
            for j in range(n_cicli):
                val = hist_list[j][i]
                med_h[i] += val
                if val < min_h[i]:
                    min_h[i] = val
                if val > max_h[i]:
                    max_h[i] = val
            med_h[i] /= n_cicli

        med_p = self.scaler.inverse_transform(pd.DataFrame(np.sqrt(med_h))).flatten().tolist()
        min_p = self.scaler.inverse_transform(pd.DataFrame(np.sqrt(min_h))).flatten().tolist()
        max_p = self.scaler.inverse_transform(pd.DataFrame(np.sqrt(max_h))).flatten().tolist()

        plt.figure(figsize=(15, 4))
        legend_parts = []
        if plot_med:
            plt.plot(med_p, label='Mean', linewidth=3.0)
            legend_parts.append('mean')
        if plot_min:
            plt.plot(min_p, label='Min')
            legend_parts.append('min')
        if plot_max:
            plt.plot(max_p, label='Max')
            legend_parts.append('max')

        plt.title(f"Values {' '.join(legend_parts)} on {n_cicli} iterations")
        plt.ylabel('Average Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if show:
            plt.show()

    # =======================================================================
    # NORMALIZATION AND TRANSFORMATIONS
    # =======================================================================

    def denormalize(self, data: Any) -> Any:
        """Scales data back to the original scale."""
        if self.scaler is None:
            raise RuntimeError("Scaler not initialized.")
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        return self.scaler.inverse_transform(df)

    @staticmethod
    def standardize(data: Any) -> Tuple[StandardScaler, pd.DataFrame]:
        """Z-score standardization (mean=0, std=1)."""
        scaler = StandardScaler()
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        scaled = scaler.fit_transform(df)
        return scaler, pd.DataFrame(scaled, columns=df.columns)

    @staticmethod
    def destandardize(data: Any, scaler: StandardScaler) -> np.ndarray:
        """Inverse of Z-score standardization."""
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        return scaler.inverse_transform(df)

    # =======================================================================
    # PRIVATE METHODS - PREPROCESSING PIPELINE
    # =======================================================================

    def _load_data(
        self,
        filepath: str,
        delimiter: str,
        date_col: str,
        drop_date: bool,
        verbose: bool
    ) -> pd.DataFrame:
        """Loads time series from CSV/TSV file."""
        data = pd.read_csv(filepath, sep=delimiter)
        if drop_date and date_col in data.columns:
            data = data.drop(columns=[date_col])
        if verbose:
            print(f"[_load_data] Loaded '{filepath}' (sep='{delimiter}'). Shape: {data.shape}")
        return data

    def _clean_data(
        self,
        data: pd.DataFrame,
        target_col: str,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        mode: str = 'drop',
        verbose: bool = False
    ) -> pd.DataFrame:
        """Cleans the time series using drop or clipping of values outside the threshold."""
        n_orig = len(data)
        if mode == 'drop':
            mask = pd.Series([True] * n_orig, index=data.index)
            if lower is not None:
                mask &= (data[target_col] >= lower)
            if upper is not None:
                mask &= (data[target_col] <= upper)
            data_clean = data[mask].copy()
        elif mode == 'clip':
            data_clean = data.copy()
            if lower is not None or upper is not None:
                data_clean[target_col] = data_clean[target_col].clip(lower=lower, upper=upper)
        else:
            raise ValueError(f"Invalid clean_mode '{mode}'. Use 'drop' or 'clip'.")

        if verbose:
            print(f"[_clean_data] mode={mode}, thresholds=[{lower}, {upper}]. "
                  f"Rows before: {n_orig}, after: {len(data_clean)} (diff: {n_orig - len(data_clean)})")
        return data_clean

    def _split_data(
        self,
        data: pd.DataFrame,
        split_value: int = 5,
        verbose: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Chronological split without shuffle in train and validation."""
        n_sample = len(data)
        n_val = n_sample // split_value
        n_train = n_sample - n_val
        data_train = data.iloc[:n_train]
        data_val = data.iloc[n_train:]
        if verbose:
            print(f"[_split_data] Total: {n_sample}, Train: {n_train}, Val: {n_val} "
                  f"({round(n_val / n_sample * 100, 2)}% val)")
        return data_train, data_val

    def _featurize(
        self,
        data: pd.DataFrame,
        n_shift: int,
        shuffle: bool = False,
        inverse: bool = False
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Sliding window: transforms a univariate series into features (x0..x_k) and target Y."""
        data_tmp = data.copy()
        col_name_str = []
        col_name_str_y = []

        for i in range(n_shift):
            data_tmp = pd.concat([data, data_tmp.shift(1)], axis=1)
            elem = f'x{i}'
            col_name_str.append(elem)
            col_name_str_y.append(elem)

        # Drop initial rows with NaN
        data_tmp = data_tmp.iloc[n_shift:].copy()
        col_name_str_y.append('Y')
        data_tmp.columns = col_name_str_y

        if inverse:
            col_name_reverse = ['Y'] + list(reversed(col_name_str))
            data_tmp = data_tmp[col_name_reverse].copy()
            data_tmp.columns = col_name_str_y

        if shuffle:
            data_tmp = data_tmp.sample(frac=1).reset_index(drop=True)

        return data_tmp, col_name_str

    def _split_xy(
        self,
        data: pd.DataFrame,
        col_name_list: List[str],
        label_col: str = 'Y'
    ) -> Tuple[Any, Any]:
        """Separates the features matrix and the target array Y."""
        x = np.asarray(data[col_name_list].values)
        y = np.asarray(data[[label_col]].values)
        return x, y

    def _normalize(
        self,
        data: pd.DataFrame,
        norm_min: float = 0.0,
        norm_max: float = 1.0,
        verbose: bool = False
    ) -> Tuple[MinMaxScaler, pd.DataFrame]:
        """MinMax normalization."""
        scaler = MinMaxScaler(feature_range=(float(norm_min), float(norm_max)))
        scaled_np = scaler.fit_transform(data)
        scaled_df = pd.DataFrame(scaled_np, columns=data.columns)
        if verbose:
            print(f"[_normalize] MinMax normalization completed in range ({norm_min}, {norm_max})")
        return scaler, scaled_df

    def _fit_batch_size(
        self,
        batch_size: int,
        t_x: np.ndarray,
        t_y: np.ndarray,
        v_x: np.ndarray,
        v_y: np.ndarray,
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Trims tensors symmetrically from the end so that the dimension is divisible by batch_size."""
        def _trim(arr: np.ndarray, bs: int) -> np.ndarray:
            """Remove the trailing samples needed for batch-size divisibility."""
            rem = arr.shape[0] % bs
            return arr if rem == 0 else arr[:-rem]

        if verbose:
            print(f"[_fit_batch_size] Initial sizes -> t_x: {t_x.shape[0]}, v_x: {v_x.shape[0]}")

        t_x_trimmed = _trim(t_x, batch_size)
        t_y_trimmed = _trim(t_y, batch_size)
        v_x_trimmed = _trim(v_x, batch_size)
        v_y_trimmed = _trim(v_y, batch_size)

        if verbose:
            print(f"[_fit_batch_size] Final sizes -> t_x: {t_x_trimmed.shape[0]}, v_x: {v_x_trimmed.shape[0]}")

        return t_x_trimmed, t_y_trimmed, v_x_trimmed, v_y_trimmed

    # =======================================================================
    # UTILITY
    # =======================================================================

    def _start_timer(self) -> float:
        """Initializes the execution timer."""
        print(time.strftime("%H:%M:%S"))
        self._timer_start = time.time()
        return self._timer_start

    def _time_elapsed(self, elapsed: float):
        """Prints elapsed time in formatted way (cross-platform, without winsound)."""
        days, rem = divmod(int(elapsed), 86400)
        hours, rem = divmod(rem, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Elapsed: {hours:02d} hh {minutes:02d} mm {seconds:02d} ss")

    def _update_progress(self, progress: float):
        """Textual progress bar."""
        bar_length = 20
        status = ""
        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1
            status = "Done...\r\n"
        block = round(bar_length * progress)
        text = f"\rProgress: [{'=' * block}{' ' * (bar_length - block)}] {round(progress * 100, 1)}% {status}"
        sys.stdout.write(text)
        sys.stdout.flush()
