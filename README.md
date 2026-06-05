# Bitcoin LSTM Prediction API

This repository contains a FastAPI-based machine learning service that predicts the next 15 minutes of Bitcoin OHLCV (Open, High, Low, Close, Volume) data based on historical 1-minute candle data. It utilizes a Structure-Aware LSTM model.

## Features
- Multi-timeframe feature engineering (1m, 15m, 1h).
- Structure-Aware LSTM inference to predict OHLCV shapes.
- Automatic reconstruction of structural predictions back into standard price values.
- Fast and easily deployable via Render.

## API Endpoint

### `POST /predict`
Accepts an array of 1-minute historical candles and returns the predicted next 15 candles.

**Note:** You must provide at least 180-200 historical 1-minute candles (roughly 3 hours of data) so the model can accurately resample to 1-hour and 15-minute intervals without losing context.

**Request Payload Example:**
```json
{
  "candles": [
    {
      "timestamp": "2026-06-04T14:00:00Z",
      "open": 63500.0,
      "high": 63550.0,
      "low": 63480.0,
      "close": 63520.0,
      "volume": 12.5
    },
    ... // at least 180 candles
  ]
}
```

**Response Example:**
```json
{
  "status": "success",
  "last_known_price": 63520.0,
  "predictions": [
    {
      "timestamp": "2026-06-04 17:01:00",
      "Open": 63521.5,
      "High": 63530.0,
      "Low": 63510.0,
      "Close": 63515.0,
      "Volume": 5.2
    },
    ... // 15 predicted candles
  ]
}
```

## Running Locally

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```
3. Test the endpoint:
   You can access the interactive Swagger UI documentation at `http://localhost:8000/docs`.

## Deployment to Render

This repository is pre-configured to be deployed as a **Blueprint** on Render.
It automatically selects the free tier.

1. Connect your GitHub repository in the [Render Dashboard](https://dashboard.render.com/) by clicking **New Blueprint Instance**.
2. Select your repository.
3. Render will use `render.yaml` to automatically provision a web service, install the dependencies using the optimized CPU-only PyTorch build, and start the `uvicorn` server.
