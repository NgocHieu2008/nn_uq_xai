# src/uq_utils.py
import torch
import numpy as np
from src.config import MC_SAMPLES

def predict_uncertainty(model, input_tensor, T=MC_SAMPLES):
    """
    Thực hiện dự đoán ngẫu nhiên T lần (MC Dropout) để tính toán độ không chắc chắn.
    """
    model.train() # Đảm bảo dropout được kích hoạt trong quá trình dự đoán
    predictions = []
    
    # Chạy T lần dự đoán
    with torch.no_grad():
        for _ in range(T):
            pred = model(input_tensor)
            predictions.append(pred.numpy())
    
    predictions = np.array(predictions) # Shape: (T, số_mẫu, 1)
    
    # Tính trung bình (Mean Prediction)
    mean_pred = predictions.mean(axis=0)
    
    # Tính độ lệch chuẩn (Uncertainty - Epistemic)
    uncertainty = predictions.std(axis=0)
    
    return mean_pred, uncertainty