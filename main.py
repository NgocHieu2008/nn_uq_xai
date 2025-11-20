# main.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
from src.config import *
from src.data_loader import get_data
from src.model import DiabetesNN

def main():
    if not os.path.exists("output/models"): os.makedirs("output/models")
 
    data = get_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DiabetesNN(input_dim=data["X_train"].shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        y_pred = model(data["X_train"].to(device))
        loss = criterion(y_pred, data["y_train"].to(device))
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f}")
            
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"-> Đã lưu model tại: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()