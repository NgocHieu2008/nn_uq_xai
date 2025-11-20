import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import DATA_PATH, TEST_SIZE, RANDOM_SEED

def get_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Lỗi: Không tìm thấy file '{DATA_PATH}'")
    
    print(f"-> Đang đọc dữ liệu từ: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    if 'Diabetes_binary' in df.columns:
        df = df.rename(columns={'Diabetes_binary': 'target'})
    
    if 'target' not in df.columns:
        raise KeyError("Lỗi: Không tìm thấy cột nhãn (target).")

    feature_names = df.drop('target', axis=1).columns.tolist()
    X = df.drop('target', axis=1).values
    y = df['target'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    data_dict = {
        "X_train": torch.FloatTensor(X_train_scaled),
        "y_train": torch.FloatTensor(y_train).unsqueeze(1),
        "X_test_tensor": torch.FloatTensor(X_test_scaled),
        "X_test_numpy": X_test,
        "X_train_numpy": X_train,
        "y_test": y_test, 
        "feature_names": feature_names,
        "scaler": scaler
    }
    
    return data_dict