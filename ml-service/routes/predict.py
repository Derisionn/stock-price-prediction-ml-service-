import pandas as pd
import torch
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List

from models.lstm_model import StructureAwareLSTM
from feature_engineering.preprocess import create_unified_dataset, reconstruct_ohlcv
from feature_engineering.dataset import CryptoMultiTimeframeDataset

router = APIRouter()

# --- Pydantic Schemas for Input Validation ---
class Candle(BaseModel):
    timestamp: str # ISO format or readable string
    open: float
    high: float
    low: float
    close: float
    volume: float

class PredictionRequest(BaseModel):
    # Require a list of candles. 120 is recommended so Pandas resample can form full 1h/15m candles
    candles: List[Candle]

@router.post("/predict")
def predict_next_15_candles(request_data: PredictionRequest, request: Request):
    """
    Accepts an array of 1m historical candles via POST, processes them, 
    and returns the predicted next 15 candles.
    """
    model = getattr(request.app.state, "model", None)
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded on the server.")
        
    if len(request_data.candles) < 60:
        raise HTTPException(status_code=400, detail="Please provide at least 60 minutes of 1m candle history.")
        
    try:
        # 1. Convert JSON payload to Pandas DataFrame
        data = [c.dict() for c in request_data.candles]
        df_1m = pd.DataFrame(data)
        
        # Capitalize columns to match our preprocessing script's expectations
        df_1m.rename(columns={
            "open": "Open", 
            "high": "High", 
            "low": "Low", 
            "close": "Close", 
            "volume": "Volume"
        }, inplace=True)
        
        df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'])
        df_1m.set_index('timestamp', inplace=True)
        df_1m.sort_index(inplace=True)
        
        # 2. Resample to 15m and 1h
        ohlcv_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
        df_15m = df_1m.resample('15min').agg(ohlcv_dict).dropna()
        df_1h = df_1m.resample('1h').agg(ohlcv_dict).dropna()
        
        # 3. Create the multi-timeframe unified dataset
        unified_df = create_unified_dataset(df_1m, df_15m, df_1h)
        
        if len(unified_df) < 60:
            raise HTTPException(
                status_code=400, 
                detail="After resampling to 1h and preventing leakage, there aren't enough valid rows. You must provide at least 180-200 candles (3 hours)."
            )
            
        dataset = CryptoMultiTimeframeDataset(unified_df, seq_length=60, pred_horizon=15)
        
        # Take the very last valid sequence
        X_latest, _ = dataset[-1]
        X_latest = X_latest.unsqueeze(0) # Shape: (1, 60, 15)
        
        # 4. Predict
        with torch.no_grad():
            preds = model(X_latest)
            
        preds = preds.squeeze(0).numpy() # Shape: (15, 5)
        pred_struct_df = pd.DataFrame(preds, columns=['open_return', 'body', 'upper_wick', 'lower_wick', 'volume_return'])
        
        # 5. Reconstruct to OHLCV
        last_known_idx = unified_df.index[-1]
        last_known_close = df_1m.loc[last_known_idx, 'Close']
        last_known_vol = df_1m.loc[last_known_idx, 'Volume']
        
        predicted_ohlcv = reconstruct_ohlcv(pred_struct_df, last_known_close, last_known_vol)
        
        # Add future timestamps
        future_times = pd.date_range(start=last_known_idx + pd.Timedelta(minutes=1), periods=15, freq='1min')
        predicted_ohlcv.index = future_times
        predicted_ohlcv.index = predicted_ohlcv.index.strftime("%Y-%m-%d %H:%M:%S")
        predicted_ohlcv.index.name = "timestamp"
        
        results = predicted_ohlcv.reset_index().to_dict(orient='records')
        
        return {
            "status": "success",
            "last_known_price": last_known_close,
            "predictions": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
