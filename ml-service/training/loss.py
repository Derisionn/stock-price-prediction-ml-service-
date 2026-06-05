import torch
import torch.nn as nn

class PINNLoss(nn.Module):
    def __init__(self, bounds, penalty_weight=1.0):
        """
        Args:
            bounds: A tensor or array of shape (5,) containing the 99.5th percentiles 
                    for the 5 structural features.
            penalty_weight: How strictly to enforce the constraints.
        """
        super(PINNLoss, self).__init__()
        self.mse = nn.MSELoss()
        # Ensure bounds is a tensor
        self.bounds = torch.tensor(bounds, dtype=torch.float32)
        self.penalty_weight = penalty_weight

    def forward(self, preds, targets):
        """
        preds shape: (batch_size, seq_length, 5)
        targets shape: (batch_size, seq_length, 5)
        Feature order: [open_return, body, upper_wick, lower_wick, volume_return]
        """
        # Base MSE Loss
        base_loss = self.mse(preds, targets)
        
        # 1. Structural Logic Constraints (Hard bounds)
        # Wicks mathematically cannot be negative because they are defined as absolute distances from High/Low
        # upper_wick = idx 2, lower_wick = idx 3
        upper_wick_preds = preds[:, :, 2]
        lower_wick_preds = preds[:, :, 3]
        
        # Penalty if wicks are negative: relu(-wick)
        wick_penalty = torch.relu(-upper_wick_preds).sum() + torch.relu(-lower_wick_preds).sum()
        
        # 2. Dynamic Data-Driven Constraints (Percentile bounds)
        # We penalize predictions that exceed the 99.5th percentile historically.
        self.bounds = self.bounds.to(preds.device)
        
        # Calculate absolute predictions
        abs_preds = torch.abs(preds)
        
        # Calculate how much the predictions exceed the bounds
        # shape of abs_preds: (batch, seq, 5)
        # shape of bounds: (5,)
        # Broadcasting handles the comparison
        exceedance = torch.relu(abs_preds - self.bounds)
        
        # Sum the exceedance across all features and sequence lengths
        bound_penalty = exceedance.sum()
        
        # Total Physics/Finance-Informed Loss
        total_loss = base_loss + (self.penalty_weight * (wick_penalty + bound_penalty))
        
        return total_loss, base_loss.item(), (wick_penalty + bound_penalty).item()

if __name__ == "__main__":
    # Test the loss function
    dummy_preds = torch.tensor([
        [[0.01, -0.02, -0.05, 0.03, 0.1]], # upper wick is negative! (anomaly)
        [[0.50, 0.01, 0.01, 0.01, 0.1]]    # open return is 50%! (exceeds bound)
    ])
    
    dummy_targets = torch.tensor([
        [[0.01, -0.02, 0.00, 0.03, 0.1]],
        [[0.00, 0.01, 0.01, 0.01, 0.1]]
    ])
    
    # Fake 99.5th percentiles bounds
    dummy_bounds = [0.05, 0.05, 0.02, 0.02, 0.5]
    
    loss_fn = PINNLoss(bounds=dummy_bounds, penalty_weight=10.0)
    
    total, mse, penalty = loss_fn(dummy_preds, dummy_targets)
    
    print(f"Total Loss: {total:.4f}")
    print(f"MSE Loss: {mse:.4f}")
    print(f"PINN Penalty: {penalty:.4f}")
    
    # Ensure penalty is non-zero because of our intentional anomalies
    if penalty > 0:
        print("SUCCESS: PINN constraints triggered on anomalous data.")
    else:
        print("ERROR: PINN constraints did not trigger.")

