import requests
import pandas as pd
import time
from datetime import datetime, timedelta

def fetch_binance_data(symbol="BTCUSDT", interval="1m", days=7):
    """
    Fetches historical candlestick data from Binance.
    """
    print(f"Fetching {days} days of {interval} data for {symbol}...")
    
    # Calculate timestamps
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)
    
    base_url = "https://api.binance.com/api/v3/klines"
    limit = 1000 # Binance max limit per request
    
    all_data = []
    
    current_start = start_time
    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": limit
        }
        
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            break
            
        all_data.extend(data)
        
        # Update current_start to the time of the last candle fetched + 1 ms
        current_start = data[-1][0] + 1
        
        # Be nice to the API
        time.sleep(0.1)

    # Define columns based on Binance API documentation
    columns = [
        "Open time", "Open", "High", "Low", "Close", "Volume",
        "Close time", "Quote asset volume", "Number of trades",
        "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
    ]
    
    df = pd.DataFrame(all_data, columns=columns)
    
    # Convert timestamps to datetime
    df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
    df["Close time"] = pd.to_datetime(df["Close time"], unit="ms")
    
    # Convert numeric columns from strings to floats
    numeric_cols = ["Open", "High", "Low", "Close", "Volume", "Quote asset volume", 
                    "Taker buy base asset volume", "Taker buy quote asset volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    
    # Drop the Ignore column as it's not needed
    df = df.drop(columns=["Ignore"])
    
    print(f"Successfully fetched {len(df)} rows.")
    return df

if __name__ == "__main__":
    # Fetch 7 days of 1-minute Bitcoin data
    df_btc = fetch_binance_data(symbol="BTCUSDT", interval="1m", days=7)
    
    # Save to a CSV file for training
    output_file = "btc_1m_7days.csv"
    df_btc.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
