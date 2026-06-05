import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys

# Add parent directory to path to import other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engineering.preprocess import load_and_resample, create_unified_dataset
from feature_engineering.dataset import CryptoMultiTimeframeDataset
from models.lstm_model import StructureAwareLSTM
from training.loss import PINNLoss

def train_phase(model, dataloader, loss_fn, optimizer, epochs=50, patience=10, phase_name="Phase 1", device="cpu"):
    """
    Standard training loop with early stopping.
    """
    model.to(device)
    loss_fn.to(device)
    
    best_loss = float('inf')
    patience_counter = 0
    
    print(f"\n--- Starting {phase_name} ---")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_penalty = 0.0
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            preds = model(batch_x)
            
            # Calculate PINN Loss
            total_loss, mse, penalty = loss_fn(preds, batch_y)
            
            # Backward pass and optimize
            total_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_mse += mse
            epoch_penalty += penalty
            
        avg_loss = epoch_loss / len(dataloader)
        avg_mse = epoch_mse / len(dataloader)
        avg_penalty = epoch_penalty / len(dataloader)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.5f} | MSE: {avg_mse:.5f} | PINN: {avg_penalty:.5f}")
        
        # Early Stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), f"best_model_{phase_name.replace(' ', '_')}.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
    # Load the best model before returning
    model.load_state_dict(torch.load(f"best_model_{phase_name.replace(' ', '_')}.pth"))
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("Loading data and creating dataset...")
    df_1m, df_15m, df_1h = load_and_resample("data/btc_1m_7days.csv")
    unified_df = create_unified_dataset(df_1m, df_15m, df_1h)
    
    # --- PHASE 1: Train to predict next 5 candles ---
    phase1_horizon = 5
    dataset_p1 = CryptoMultiTimeframeDataset(unified_df, seq_length=60, pred_horizon=phase1_horizon)
    dataloader_p1 = DataLoader(dataset_p1, batch_size=64, shuffle=True)
    
    bounds = dataset_p1.get_percentile_bounds()
    print(f"PINN Constraint Bounds (99.5th percentile): {bounds}")
    
    model = StructureAwareLSTM(input_dim=15, hidden_dim=128, num_layers=2, pred_horizon=phase1_horizon)
    loss_fn = PINNLoss(bounds=bounds, penalty_weight=2.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # L2 Regularization included
    
    # Train Phase 1
    # Note: Setting epochs=3 for quick testing. In real life, use 50-100.
    model = train_phase(model, dataloader_p1, loss_fn, optimizer, epochs=3, patience=2, phase_name="Phase 1", device=device)
    
    # --- PHASE 2: Train to predict next 15 candles (Curriculum Learning) ---
    phase2_horizon = 15
    dataset_p2 = CryptoMultiTimeframeDataset(unified_df, seq_length=60, pred_horizon=phase2_horizon)
    dataloader_p2 = DataLoader(dataset_p2, batch_size=64, shuffle=True)
    
    # Change model head for Phase 2 (freezes LSTM layers)
    print("\nAdapting model head for Phase 2 curriculum learning...")
    model.change_prediction_horizon(new_horizon=phase2_horizon)
    
    # Create new optimizer that only updates the new fully connected head
    optimizer_p2 = optim.Adam(model.fc.parameters(), lr=0.001)
    
    # Train Phase 2
    model = train_phase(model, dataloader_p2, loss_fn, optimizer_p2, epochs=3, patience=2, phase_name="Phase 2", device=device)
    
    print("\nTraining Complete! Model successfully learned via multi-timeframe curriculum learning.")

