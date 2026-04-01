import torch.nn as nn


class RNetV1Head(nn.Module):
    """
    Lightweight MLP head that predicts rot6d residuals for local joints.
    """

    def __init__(self, input_dim, hidden_dim=256, output_dim=6, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # x: [B, T, J, F]
        return self.mlp(x)
