import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from ta.momentum import RSIIndicator
from ta.trend import MACD
import warnings
from tqdm import tqdm
from numba import jit, prange

def load_and_process_data(filepath: str, debug: bool = False) -> pd.DataFrame:
    """Loads CSV file, sets index, and converts datetime."""
    try:
        df = pd.read_csv(filepath, index_col="Gmt time")
        df.index = pd.to_datetime(df.index, format="%d.%m.%Y %H:%M:%S.%f")
        if debug:
            print(f"Successfully loaded: {filepath}")
        return df.dropna()
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return pd.DataFrame()

def process_test_data(datapath1: str, datapath2: str, window: int = 60, min_std_threshold: float = 1e-6, debug: bool = False):
    """
    Loads and aligns two datasets by common timestamps, and filters out flat periods.

    Args:
        datapath1: Path to first CSV file
        datapath2: Path to second CSV file
        window: Rolling window size for standard deviation calculation
        min_std_threshold: Minimum standard deviation threshold to consider price movement significant
        debug: Whether to print debug information

    Returns:
        Tuple of aligned DataFrames with flat periods removed
    """
    symbol1, symbol2 = Path(datapath1).stem, Path(datapath2).stem
    data1, data2 = load_and_process_data(datapath1, debug), load_and_process_data(datapath2, debug)

    if data1.empty or data2.empty:
        print("One or both datasets failed to load.")
        return None, None

    # Step 1: Find common timestamps first
    common_index = data1.index.intersection(data2.index)
    data1 = data1.reindex(common_index)
    data2 = data2.reindex(common_index)

    if debug:
        print(f"Common timestamps: {len(common_index)}")
        print(f"Start date: {common_index.min()}, End date: {common_index.max()}")

    # Step 2: Calculate rolling standard deviations
    # Use NumPy arrays for faster computation
    close1 = data1['Close'].values
    close2 = data2['Close'].values
    std1_values = np.full(len(close1), np.nan)
    std2_values = np.full(len(close2), np.nan)

    # Calculate standard deviations with JIT-optimized function
    calculate_rolling_std(close1, std1_values, window)
    calculate_rolling_std(close2, std2_values, window)

    std1 = pd.Series(std1_values, index=data1.index)
    std2 = pd.Series(std2_values, index=data2.index)

    # Step 3: Identify periods with sufficient volatility
    valid_indices = (std1 > min_std_threshold) & (std2 > min_std_threshold)

    # Step 4: Filter out weekends (optional)
    is_weekday = pd.Series(~np.isin(data1.index.dayofweek, [5, 6]), index=data1.index)
    valid_indices = valid_indices & is_weekday

    # Step 5: Apply filters
    filtered_data1 = data1[valid_indices]
    filtered_data2 = data2[valid_indices]

    if debug:
        removed_pct = (1 - len(filtered_data1) / len(data1)) * 100
        print(f"Removed {removed_pct:.2f}% of data points as flat or weekend periods")
        print(f"Remaining data points: {len(filtered_data1)}")

        # Verify alignment
        if not filtered_data1.index.equals(filtered_data2.index):
            print("WARNING: Indices are not aligned after filtering!")
        else:
            print("Indices are perfectly aligned.")
        print("Data Preprocessing Successful")

    # Step 6: Return the aligned dataframes
    return filtered_data1, filtered_data2

@jit(nopython=True)
def calculate_rolling_std(arr, output, window):
    """
    Calculate rolling standard deviation with Numba acceleration.
    """
    n = len(arr)
    for i in range(n - window + 1):
        output[i + window - 1] = np.std(arr[i:i + window])
    return output

@jit(nopython=True)
def calc_lstsq(X, y):
    """
    Simplified least squares calculation for Numba.
    """
    # Solve X'X b = X'y
    XTX = X.T @ X
    XTy = X.T @ y

    # Check if XTX is invertible
    det = XTX[0, 0] * XTX[1, 1] - XTX[0, 1] * XTX[1, 0]
    if abs(det) < 1e-10:
        return np.array([np.nan, np.nan])

    # Inverse of 2x2 matrix
    inv_XTX = np.empty((2, 2))
    inv_XTX[0, 0] = XTX[1, 1] / det
    inv_XTX[1, 1] = XTX[0, 0] / det
    inv_XTX[0, 1] = -XTX[0, 1] / det
    inv_XTX[1, 0] = -XTX[1, 0] / det

    # Calculate beta
    beta = inv_XTX @ XTy
    return beta

@jit(nopython=True)
def rolling_cointegration_score(asset1, asset2, window=60, eps=1e-10):
    """
    Computes a rolling cointegration score using simplified ADF test logic.
    Optimized for Numba.

    Args:
        asset1: First asset price series as numpy array
        asset2: Second asset price series as numpy array
        window: Rolling window size
        eps: Small epsilon value to prevent division by zero
    """
    n = len(asset1)
    if n < window:
        raise ValueError("Window size too large for available data")

    scores = np.full(n, np.nan)

    for i in range(n - window + 1):
        y = asset1[i:i + window]
        x_vals = asset2[i:i + window]

        # Check if there's enough variation in both series
        y_std = np.std(y)
        x_std = np.std(x_vals)

        if y_std < eps or x_std < eps:
            continue

        # Create design matrix
        X = np.column_stack((np.ones(window), x_vals))

        # Calculate coefficients
        beta = calc_lstsq(X, y)

        # If calculation failed, skip
        if np.isnan(beta[0]):
            continue

        # Calculate residuals
        residuals = y - (beta[0] + beta[1] * x_vals)

        # Check if residuals are essentially flat
        if np.std(residuals) < eps:
            continue

        # Simplified ADF test statistic (using residual variance)
        # This is a proxy for the ADF test since full ADF test is complex for numba
        adf_stat = -np.std(residuals) / np.mean(np.abs(residuals))
        scores[i + window - 1] = adf_stat

    return scores

def process_train_data(asset1: pd.DataFrame, asset2: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """Computes training features from asset data."""
    if debug:
        print("Processing data for Neural Network training.")
    df = pd.DataFrame(index=asset1.index)
    df["Asset1_Close"], df["Asset2_Close"] = asset1["Close"], asset2["Close"]

    spread = df["Asset1_Close"] - df["Asset2_Close"]

    df["Rolling Spread-ZScore"] = (spread - spread.rolling(30).mean()) / spread.rolling(30).std()
    df["Rolling Correlation"] = df["Asset1_Close"].rolling(30).corr(df["Asset2_Close"])

    # Run cointegration score calculation
    coint_scores = rolling_cointegration_score(
        df["Asset1_Close"].values,
        df["Asset2_Close"].values
    )
    df["Rolling Cointegration Score"] = coint_scores

    # Add technical indicators
    df["RSI1"] = RSIIndicator(df["Asset1_Close"], window=14).rsi()
    df["RSI2"] = RSIIndicator(df["Asset2_Close"], window=14).rsi()
    df["MACD1"] = MACD(df["Asset1_Close"], window_slow=26, window_fast=12, window_sign=9).macd()
    df["MACD2"] = MACD(df["Asset2_Close"], window_slow=26, window_fast=12, window_sign=9).macd()

    df.dropna(inplace=True)
    df.to_csv("./data/processed/train_data.csv")
    return df
