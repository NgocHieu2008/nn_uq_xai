# src/model.py
import torch.nn as nn
from src.config import DROPOUT_RATE

class DiabetesNN(nn.Module):
    def __init__(self, input_dim):
        super(DiabetesNN, self).__init__()
        
        # Layer 1: Input -> 64
        self.layer1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)   
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=DROPOUT_RATE)
        
        # Layer 2: 64 -> 32
        self.layer2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)   
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=DROPOUT_RATE)
        
        # Output: 32 -> 1
        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Thứ tự chuẩn: Linear -> Batch Norm -> Activation -> Dropout
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        
        return self.sigmoid(self.output(x))