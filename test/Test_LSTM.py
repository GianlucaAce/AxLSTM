"""
Test_LSTM_train.py
=================
Script to train and evaluate time series using
the AxLSTM class.
- Use of unidirectional LSTM (forecasting)
- Multi-layer architecture with configurable units (default: [64, 32])
- Addition of Dropout (0.2) between layers for regularization
- EarlyStopping (patience=10)
- Parametrization via command line arguments (dataset path, hyperparameters)
"""

import os
import sys
import argparse
from typing import List, Optional

# Suppressing Informational Messages and TensorFlow's oneDNN C++
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from ax_lstm import AxLSTM

def resolve_data_path(custom_path: Optional[str] = None) -> str:
    """Resolves the data file path by checking known locations."""
    if custom_path and os.path.exists(custom_path):
        return os.path.abspath(custom_path)

    candidate_paths = [
        custom_path,
        os.path.join(SCRIPT_DIR, 'Test.txt'),
        os.path.join(SCRIPT_DIR, 'data', 'Test.txt'),
        os.path.join(SCRIPT_DIR, 'Data', 'Test.txt'),
        os.path.join(SCRIPT_DIR, 'test', 'Test.txt'),
        os.path.join(SCRIPT_DIR, 'Test', 'Test.txt'),
        os.path.join(SCRIPT_DIR, '..', 'Test', 'Data', 'Test.txt'),
        os.path.join(SCRIPT_DIR, '..', 'Test', 'data', 'Test.txt'),
        'data/Test.txt',
        'test/Test.txt',
        'Data/Test.txt',
        '../Test/Data/Test.txt',
    ]

    for path in candidate_paths:
        if path and os.path.exists(path):
            return os.path.abspath(path)

    searched_paths = [path for path in candidate_paths if path]
    raise FileNotFoundError(
        f"Data file not found. Searched: {', '.join(searched_paths)}"
    )


def parse_args():
    """Definition and parsing of command line arguments."""
    parser = argparse.ArgumentParser(
        description="Training LSTM network for time series with AxLSTM"
    )

    # Dataset parameters
    parser.add_argument(
        '--data', type=str, default=None,
        help="Path to the time series CSV/TSV file (default: auto-detect Test.txt)"
    )
    parser.add_argument(
        '--delimiter', type=str, default='\t',
        help="Column delimiter in data file (default: tab '\\t')"
    )
    parser.add_argument(
        '--target-col', type=str, default='y',
        help="Target column name (default: 'y')"
    )
    parser.add_argument(
        '--date-col', type=str, default='data',
        help="Column name data to be discarded if present (default: 'data')"
    )
    parser.add_argument(
        '--clean-mode', type=str, choices=['drop', 'clip', 'none'], default='clip',
        help="Outlier cleaning mode: 'clip' (as in notebook), 'drop' or 'none'"
    )
    parser.add_argument(
        '--clean-lower', type=float, default=250.0,
        help="Lower bound for data cleaning (default: 250.0)"
    )
    parser.add_argument(
        '--clean-upper', type=float, default=1950.0,
        help="Upper bound for data cleaning (default: 1950.0)"
    )

    # Model hyperparameters
    parser.add_argument(
        '--time-steps', type=int, default=7,
        help="Sliding window size (default: 7)"
    )
    parser.add_argument(
        '--split-value', type=int, default=5,
        help="Fraction for validation set split: 1/N of total (default: 5 -> 20%%)"
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        '--units', type=int, nargs='+', default=[64, 32],
        help="Number of units for each LSTM layer (default: 64 32)"
    )
    parser.add_argument(
        '--dropout', type=float, default=0.2,
        help="Dropout rate between LSTM layers (default: 0.2)"
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.001,
        help="Learning rate for Adam optimizer (default: 0.001)"
    )
    parser.add_argument(
        '--stateful', action='store_true', default=False,
        help="If specified, enables LSTM stateful mode (default: False)"
    )
    parser.add_argument(
        '--epochs', type=int, default=100,
        help="Maximum number of training epochs (default: 100)"
    )
    parser.add_argument(
        '--patience', type=int, default=10,
        help="Patience for EarlyStopping on val_loss (default: 10, 0 to disable)"
    )

    # Visualization
    parser.add_argument(
        '--no-plot', action='store_true', default=False,
        help="If specified, disables final graph visualization"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    data_path = resolve_data_path(args.data)

    print("=" * 70)
    print("Starting AxLSTM Pipeline - Time Series Training")
    print("=" * 70)
    print(f"Data file:         {data_path}")
    print(f"Delimiter:         {repr(args.delimiter)}")
    print(f"Data cleaning:     mode={args.clean_mode} (lower={args.clean_lower}, upper={args.clean_upper})")
    print(f"Time steps:        {args.time_steps}")
    print(f"Batch size:        {args.batch_size}")
    print(f"Units LSTM:        {args.units}")
    print(f"Dropout:           {args.dropout}")
    print(f"Learning rate:     {args.learning_rate}")
    print(f"Stateful:          {args.stateful}")
    print(f"Max epochs:        {args.epochs}")
    print(f"EarlyStopping pat: {args.patience}")
    print("=" * 70)

    # 1. Hyperparameter configuration
    config = {
        'time_steps': args.time_steps,
        'split_value': args.split_value,
        'batch_size': args.batch_size,
        'units': args.units,
        'dropout': args.dropout,
        'learning_rate': args.learning_rate,
        'stateful': args.stateful,
        'early_stopping_patience': args.patience,
    }

    # 2. Initialize AxLSTM object
    model = AxLSTM(config)

    # 3. Load and prepare data (unique and clean pipeline)
    print("\n[1/4] Loading and preparing data...")
    model.load_and_prepare(
        filepath=data_path,
        delimiter=args.delimiter,
        target_col=args.target_col,
        date_col=args.date_col,
        drop_date=True,
        clean_mode=args.clean_mode,
        clean_lower=args.clean_lower,
        clean_upper=args.clean_upper,
        verbose=True
    )

    # 4. Build Keras model
    print("\n[2/4] Building LSTM model...")
    model.build_model(verbose=True)

    # 5. Training with EarlyStopping
    print("\n[3/4] Training model...")
    model.train(
        epochs=args.epochs,
        shuffle=not args.stateful,
        verbose=1
    )

    # 6. Evaluation
    print("\n[4/4] Evaluating model...")
    metrics = model.evaluate()
    print(f"Final results -> Train MSE: {metrics['train_loss']:.6f}, Val MSE: {metrics['val_loss']:.6f}")

    # 7. Graphs (if not disabled)
    if not args.no_plot:
        print("\nGenerating graphs (unified dashboard)...")
        model.plot_dashboard(
            title_history='Loss Trend (Denormalized)',
            title_predictions='Prediction vs Real Values',
            y_label='Value',
            x_label='Samples'
        )

    print("\n[Pipeline completed successfully]")

if __name__ == '__main__':
    main()