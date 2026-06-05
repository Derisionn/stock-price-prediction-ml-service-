import torch
import torch.nn as nn

class StructureAwareLSTM(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=128, num_layers=2, pred_horizon=5):
        """
        Args:
            input_dim: 15 (5 features * 3 timeframes)
            hidden_dim: Number of LSTM units per layer
            num_layers: Number of stacked LSTM layers
            pred_horizon: Number of future steps to predict (e.g., 5 or 15)
        """
        super(StructureAwareLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_horizon = pred_horizon
        
        # The LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # The final dense head
        # We need to predict 5 features for each future timestep in the horizon
        self.fc = nn.Linear(hidden_dim, pred_horizon * 5)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # We only care about the output from the LAST timestep in the sequence
        out = out[:, -1, :]
        
        # Pass through the linear head
        out = self.fc(out)
        
        # Reshape to (batch_size, pred_horizon, 5)
        out = out.view(-1, self.pred_horizon, 5)
        return out

    def change_prediction_horizon(self, new_horizon):
        """
        Used for Curriculum Learning. Freezes the LSTM layers and replaces 
        the linear head to predict a new horizon (e.g., changing from 5 to 15).
        """
        # Freeze LSTM layers
        for param in self.lstm.parameters():
            param.requires_grad = False
            
        self.pred_horizon = new_horizon
        # Replace the fully connected layer
        self.fc = nn.Linear(self.hidden_dim, new_horizon * 5).to(next(self.parameters()).device)

if __name__ == "__main__":
    # Test the model
    model = StructureAwareLSTM(pred_horizon=5)
    
    # Dummy input: batch_size=32, seq_length=60, input_dim=15
    dummy_x = torch.randn(32, 60, 15)
    
    # Test Phase 1 forward pass
    out_5 = model(dummy_x)
    print(f"Phase 1 (horizon=5) output shape: {out_5.shape}")  # Expected: [32, 5, 5]
    
    # Test changing horizon for Phase 2
    model.change_prediction_horizon(new_horizon=15)
    out_15 = model(dummy_x)
    print(f"Phase 2 (horizon=15) output shape: {out_15.shape}")  # Expected: [32, 15, 5]
    
    # Verify LSTM is frozen
    lstm_frozen = all(not p.requires_grad for p in model.lstm.parameters())
    print(f"LSTM layers frozen for curriculum learning: {lstm_frozen}")

