import pandas as pd
import numpy as np

def load_and_resample(file_path):
    """Loads 1m data and returns resampled 1m, 15m, and 1h dataframes."""
    df_1m = pd.read_csv(file_path)
    df_1m['Open time'] = pd.to_datetime(df_1m['Open time'])
    df_1m.set_index('Open time', inplace=True)
    df_1m.sort_index(inplace=True)
    
    # Define aggregation logic for candlesticks
    ohlcv_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    
    df_15m = df_1m.resample('15min').agg(ohlcv_dict).dropna()
    df_1h = df_1m.resample('1h').agg(ohlcv_dict).dropna()
    
    return df_1m, df_15m, df_1h

def compute_structural_features(df):
    """
    Transforms raw OHLCV into stationary structural percentages.
    Returns a DataFrame with the structural features.
    """
    df_struct = pd.DataFrame(index=df.index)
    
    # 1. Open Return: (Open[t] - Close[t-1]) / Close[t-1]
    df_struct['open_return'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    
    # 2. Body Size: (Close[t] - Open[t]) / Open[t]
    df_struct['body'] = (df['Close'] - df['Open']) / df['Open']
    
    # 3. Upper Wick: (High[t] - max(Open[t], Close[t])) / Open[t]
    max_open_close = df[['Open', 'Close']].max(axis=1)
    df_struct['upper_wick'] = (df['High'] - max_open_close) / df['Open']
    
    # 4. Lower Wick: (min(Open[t], Close[t]) - Low[t]) / Open[t]
    min_open_close = df[['Open', 'Close']].min(axis=1)
    df_struct['lower_wick'] = (min_open_close - df['Low']) / df['Open']
    
    # 5. Volume Return: (Volume[t] - Volume[t-1]) / (Volume[t-1] + epsilon)
    # Using log return for volume is often more stable, or adding epsilon
    epsilon = 1e-8
    df_struct['volume_return'] = (df['Volume'] - df['Volume'].shift(1)) / (df['Volume'].shift(1) + epsilon)
    
    # Drop the first row which will have NaN for returns
    return df_struct.dropna()

def reconstruct_ohlcv(df_struct, initial_close, initial_volume):
    """
    Reconstructs raw OHLCV from structural features, given the previous close and volume.
    Used for inference and testing.
    """
    reconstructed = pd.DataFrame(index=df_struct.index, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    prev_close = initial_close
    prev_vol = initial_volume
    
    for idx, row in df_struct.iterrows():
        # Open
        curr_open = prev_close * (1 + row['open_return'])
        
        # Close
        curr_close = curr_open * (1 + row['body'])
        
        # Wicks
        max_oc = max(curr_open, curr_close)
        min_oc = min(curr_open, curr_close)
        
        curr_high = max_oc + (row['upper_wick'] * curr_open)
        curr_low = min_oc - (row['lower_wick'] * curr_open)
        
        # Volume
        curr_vol = prev_vol * (1 + row['volume_return'])
        
        reconstructed.loc[idx] = [curr_open, curr_high, curr_low, curr_close, curr_vol]
        
        prev_close = curr_close
        prev_vol = curr_vol
        
    # Convert types back to float
    reconstructed = reconstructed.astype(float)
    return reconstructed

def create_unified_dataset(df_1m, df_15m, df_1h):
    """
    Computes structural features for all timeframes, aligns them, 
    and prevents data leakage by shifting the higher timeframes.
    """
    struct_1m = compute_structural_features(df_1m)
    struct_15m = compute_structural_features(df_15m)
    struct_1h = compute_structural_features(df_1h)
    
    # Add suffixes to distinguish timeframes
    struct_1m.columns = [f"{col}_1m" for col in struct_1m.columns]
    struct_15m.columns = [f"{col}_15m" for col in struct_15m.columns]
    struct_1h.columns = [f"{col}_1h" for col in struct_1h.columns]
    
    # Shift higher timeframes by 1 to prevent data leakage!
    # A 10:00 15-minute candle contains data from 10:00 to 10:14:59.
    # It should only be visible to 1m candles starting at 10:15.
    struct_15m = struct_15m.shift(1)
    struct_1h = struct_1h.shift(1)
    
    # Merge using forward fill
    # We join everything on the 1m index.
    unified = struct_1m.copy()
    
    # Reindex higher timeframes to match the 1m index, then forward fill
    struct_15m_aligned = struct_15m.reindex(unified.index).ffill()
    struct_1h_aligned = struct_1h.reindex(unified.index).ffill()
    
    unified = unified.join(struct_15m_aligned)
    unified = unified.join(struct_1h_aligned)
    
    # Drop rows that have NaNs (the very beginning of the dataset)
    return unified.dropna()

if __name__ == "__main__":
    file_path = "data/btc_1m_7days.csv"
    print("Loading and resampling data...")
    df_1m, df_15m, df_1h = load_and_resample(file_path)
    
    print("Testing reconstruction logic...")
    # Test on a small subset (e.g. first 100 rows after the first shifted row)
    df_1m_struct = compute_structural_features(df_1m)
    subset_orig = df_1m.iloc[1:101].copy()
    subset_struct = df_1m_struct.iloc[0:100].copy()
    
    initial_close = df_1m.iloc[0]['Close']
    initial_vol = df_1m.iloc[0]['Volume']
    
    reconstructed = reconstruct_ohlcv(subset_struct, initial_close, initial_vol)
    
    diff = np.abs(subset_orig[['Open', 'High', 'Low', 'Close', 'Volume']].values - reconstructed.values)
    max_diff = np.max(diff)
    
    if max_diff < 1e-4:
        print(f"SUCCESS! Reconstruction is perfect. (Max diff: {max_diff})")
    else:
        print(f"ERROR: Reconstruction failed. (Max diff: {max_diff})")
        
    print("Creating unified multi-timeframe dataset...")
    unified_df = create_unified_dataset(df_1m, df_15m, df_1h)
    print(f"Unified dataset shape: {unified_df.shape}")
    print("Columns:", list(unified_df.columns))

