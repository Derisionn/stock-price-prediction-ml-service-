import torch
from fastapi import FastAPI
import uvicorn
import os
import sys

# Ensure Python can find our modules inside the ml-service folder
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml-service"))

from models.lstm_model import StructureAwareLSTM
from routes.predict import router as predict_router

app = FastAPI(title="Bitcoin LSTM Prediction API", description="Predicts the next 15 minutes of BTC OHLCV")

# Global variables
MODEL_PATH = "best_model_Phase_2.pth"

@app.on_event("startup")
def load_model():
    """Loads the trained LSTM model into memory when the server starts."""
    try:
        print("Loading trained Phase 2 LSTM Model...")
        model = StructureAwareLSTM(input_dim=15, hidden_dim=128, num_layers=2, pred_horizon=15)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        # Store the model on the app state so routers can access it
        app.state.model = model
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        app.state.model = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the Bitcoin LSTM Prediction API. Send a POST request to /predict."}

# Include the routes from ml-service/routes/
app.include_router(predict_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
