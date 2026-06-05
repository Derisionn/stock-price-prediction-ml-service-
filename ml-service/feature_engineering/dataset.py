import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CryptoMultiTimeframeDataset(Dataset):
    def __init__(self, unified_df, seq_length=60, pred_horizon=5):
        """
        Args:
            unified_df: DataFrame containing the 1m, 15m, and 1h aligned features.
            seq_length: Number of past timesteps to look back (default 60 mins).
            pred_horizon: Number of future 1m timesteps to predict (default 5 for Phase 1).
        """
        self.seq_length = seq_length
        self.pred_horizon = pred_horizon
        
        # Convert to numpy arrays for speed
        self.data = unified_df.values.astype(np.float32)
        
        # The target features are the first 5 columns (the 1m structural features)
        # Columns: open_return_1m, body_1m, upper_wick_1m, lower_wick_1m, volume_return_1m
        self.targets = unified_df.iloc[:, :5].values.astype(np.float32)
        
        # We can only create samples where we have enough history AND enough future
        self.valid_indices = np.arange(self.seq_length, len(self.data) - self.pred_horizon)
        
        # Calculate dynamic bounds (e.g., 99.5th percentiles) for the PINN constraints
        self.percentiles_995 = np.percentile(np.abs(self.targets), 99.5, axis=0)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # The actual index in the dataframe
        end_idx = self.valid_indices[idx]
        
        # X shape: (seq_length, 15) -> 60 minutes of all multi-timeframe features
        X = self.data[end_idx - self.seq_length : end_idx]
        
        # y shape: (pred_horizon, 5) -> The next 'pred_horizon' minutes of 1m structural features
        y = self.targets[end_idx : end_idx + self.pred_horizon]
        
        return torch.tensor(X), torch.tensor(y)

    def get_percentile_bounds(self):
        """Returns the 99.5th percentile values for [open_ret, body, upper, lower, vol_ret]"""
        return self.percentiles_995


if __name__ == "__main__":
    from preprocess import load_and_resample, create_unified_dataset
    
    # Quick test
    df_1m, df_15m, df_1h = load_and_resample("../data/btc_1m_7days.csv")
    unified_df = create_unified_dataset(df_1m, df_15m, df_1h)
    
    # Create dataset for Phase 1 (predicting 5 candles)
    dataset = CryptoMultiTimeframeDataset(unified_df, seq_length=60, pred_horizon=5)
    
    print(f"Dataset length: {len(dataset)}")
    
    X_sample, y_sample = dataset[0]
    print(f"X shape: {X_sample.shape}")
    print(f"y shape: {y_sample.shape}")
    
    bounds = dataset.get_percentile_bounds()
    print("99.5th Percentile Bounds for PINN Loss constraints:")
    features = ['open_return', 'body', 'upper_wick', 'lower_wick', 'volume_return']
    for f, b in zip(features, bounds):
        print(f"  {f}: {b:.5f}")

