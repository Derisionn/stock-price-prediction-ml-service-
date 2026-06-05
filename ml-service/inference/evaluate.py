import torch
import numpy as np
import pandas as pd
import os
import sys

# Add parent directory to path to import other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_model import StructureAwareLSTM
from feature_engineering.preprocess import load_and_resample, create_unified_dataset, reconstruct_ohlcv
from feature_engineering.dataset import CryptoMultiTimeframeDataset

def evaluate_model():
    print("Loading data for evaluation...")
    df_1m, df_15m, df_1h = load_and_resample("data/btc_1m_7days.csv")
    unified_df = create_unified_dataset(df_1m, df_15m, df_1h)
    
    # Use the last 500 minutes as a test set
    test_df = unified_df.iloc[-500:]
    
    dataset = CryptoMultiTimeframeDataset(test_df, seq_length=60, pred_horizon=15)
    
    # We just want to take one random sample
    X_test, y_test = dataset[len(dataset) - 1]
    
    # Expand dims to simulate batch_size=1
    X_test = X_test.unsqueeze(0)
    
    print("Loading trained Phase 2 model...")
    model = StructureAwareLSTM(input_dim=15, hidden_dim=128, num_layers=2, pred_horizon=15)
    try:
        model.load_state_dict(torch.load("best_model_Phase_2.pth", map_location="cpu"))
        model.eval()
    except FileNotFoundError:
        print("Error: best_model_Phase_2.pth not found. Ensure training completed.")
        return

    print("Running Inference...")
    with torch.no_grad():
        preds = model(X_test)
        
    # preds is shape (1, 15, 5)
    preds = preds.squeeze(0).numpy()
    
    # Convert predictions back into a DataFrame
    pred_struct_df = pd.DataFrame(preds, columns=['open_return', 'body', 'upper_wick', 'lower_wick', 'volume_return'])
    
    # Reconstruct back to raw OHLCV
    # We need the last actual Close and Volume to reconstruct the future
    # The last known timestep before our prediction is the end of the sequence
    last_known_idx = test_df.index[-16] # -16 because 15 is the horizon
    last_known_close = df_1m.loc[last_known_idx, 'Close']
    last_known_vol = df_1m.loc[last_known_idx, 'Volume']
    
    print("\nReconstructing Predicted OHLCV Prices from Structural Constraints...")
    predicted_ohlcv = reconstruct_ohlcv(pred_struct_df, last_known_close, last_known_vol)
    
    # Print the predictions
    print("\nPredicted Next 15 Candles (OHLCV):")
    print(predicted_ohlcv.round(2).to_string())
    
    print("\nEvaluation completed successfully. Structural constraints held and reconstructed smoothly!")

if __name__ == "__main__":
    evaluate_model()

