# run_analysis.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lime
from lime import lime_tabular
import shap
import os
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.calibration import calibration_curve # <--- Mới
from datetime import datetime

from src.config import *
from src.data_loader import get_data
from src.model import DiabetesNN
from src.uq_utils import predict_uncertainty

plt.style.use('ggplot')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(directory):
    if not os.path.exists(directory): os.makedirs(directory)

def plot_advanced_metrics(y_true, y_probs, uncertainties, save_dir):
    """Vẽ Calibration Curve và Rejection Curve"""
    print("   -> Vẽ biểu đồ Calibration và Rejection...")
    
    # --- 1. Calibration Curve ---
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10, strategy='uniform')
    plt.figure(figsize=(8, 8))
    plt.plot(prob_pred, prob_true, marker='o', label='Model Calibration', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Reliability Diagram (Calibration Curve)')
    plt.legend()
    plt.savefig(os.path.join(save_dir, "4_calibration_curve.png"), dpi=300)
    plt.close()

    # --- 2. Rejection Curve ---
    # Sắp xếp theo độ không chắc chắn tăng dần (tự tin nhất -> kém tự tin nhất)
    sorted_indices = np.argsort(uncertainties)
    y_true_sorted = y_true[sorted_indices]
    y_pred_label_sorted = (y_probs[sorted_indices] > 0.5).astype(int)
    
    fractions = np.linspace(0, 0.5, 20) # Loại bỏ từ 0% đến 50% dữ liệu xấu nhất
    accuracies = []
    total_samples = len(y_true)
    
    for frac in fractions:
        n_keep = int(total_samples * (1 - frac))
        if n_keep < 10: break
        acc = accuracy_score(y_true_sorted[:n_keep], y_pred_label_sorted[:n_keep])
        accuracies.append(acc)
        
    plt.figure(figsize=(10, 6))
    plt.plot(fractions * 100, accuracies, marker='s', color='green', linewidth=2)
    plt.xlabel('Percentage of Uncertain Samples Rejected (%)')
    plt.ylabel('Accuracy on Remaining Data')
    plt.title('Error Retention / Rejection Curve')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "5_rejection_curve.png"), dpi=300)
    plt.close()

def analyze_single_case(case_idx, data_dict, model, explainer, title_prefix, save_dir):
    patient_raw = data_dict["X_test_numpy"][case_idx]
    def predict_fn_lime(numpy_data):
        scaled_data = data_dict["scaler"].transform(numpy_data)
        tensor_data = torch.FloatTensor(scaled_data)
        model.eval()
        with torch.no_grad():
            prob_pos = model(tensor_data).numpy()
        return np.hstack((1 - prob_pos, prob_pos))

    print(f"      -> Vẽ LIME cho ca {case_idx} ({title_prefix})...")
    exp = explainer.explain_instance(patient_raw, predict_fn_lime, num_features=10)
    fig = exp.as_pyplot_figure()
    plt.title(f"{title_prefix} (Index: {case_idx})")
    plt.tight_layout()
    filename = f"lime_{title_prefix.replace(' ', '_')}_idx{case_idx}.png"
    fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("output", "figures", f"run_{timestamp}")
    ensure_dir(run_dir)
    print(f"=== PHÂN TÍCH TOÀN DIỆN (ID: {timestamp}) ===")

    # 1. Load
    print("\n[1/5] Tải dữ liệu & Model...")
    data = get_data()
    model = DiabetesNN(input_dim=data["X_train"].shape[1]).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print("   -> Model loaded.")
    except FileNotFoundError:
        print("   -> LỖI: Hãy chạy 'python main.py' trước.")
        return

    # 2. Predict
    print("\n[2/5] Tính toán UQ...")
    model.cpu()
    mean_preds, uncertainties = predict_uncertainty(model, data["X_test_tensor"])
    
    y_true = data["y_test"]
    y_pred_label = (mean_preds > 0.5).astype(int).flatten()
    mean_preds_flat = mean_preds.flatten()
    uncertainties_flat = uncertainties.flatten()

    # --- Biểu đồ 1: Histogram ---
    plt.figure(figsize=(10, 6))
    sns.histplot(uncertainties_flat, kde=True, bins=30, color="skyblue")
    plt.title("Uncertainty Distribution")
    plt.axvline(np.mean(uncertainties_flat), color='r', linestyle='--')
    plt.savefig(os.path.join(run_dir, "1_distribution.png"), dpi=300)
    plt.close()

    # --- Biểu đồ 2: Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred_label)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(os.path.join(run_dir, "2_confusion_matrix.png"), dpi=300)
    plt.close()

    # --- Biểu đồ 4 & 5: Advanced Metrics (Calibration & Rejection) ---
    plot_advanced_metrics(y_true, mean_preds_flat, uncertainties_flat, run_dir)

    # 3. Stats
    auc = roc_auc_score(y_true, mean_preds_flat)
    acc = accuracy_score(y_true, y_pred_label)
    unc_threshold = np.percentile(uncertainties_flat, 90)
    
    stats_file = os.path.join(run_dir, "report.txt")
    with open(stats_file, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nAUC: {auc:.4f}\n")
        f.write(f"Uncertain threshold (Top 10%): {unc_threshold:.4f}\n")
    print(f"   -> Accuracy: {acc:.4f} | AUC: {auc:.4f}")

    # 4. SHAP (Global)
    print("\n[4/5] SHAP Analysis...")
    background = data["X_train"][:200]
    test_samples_shap = data["X_test_tensor"][:50]
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(test_samples_shap)
    
    if isinstance(shap_values, list): shap_values = shap_values[0]
    if len(shap_values.shape) == 3: shap_values = shap_values.squeeze()

    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, test_samples_shap.numpy(), feature_names=data["feature_names"], plot_type="dot", show=False, max_display=25)
    plt.savefig(os.path.join(run_dir, "3_shap_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. LIME (Local Case Studies)
    print("\n[5/5] LIME Case Studies...")
    idx_most_unc = np.argmax(uncertainties_flat)
    fp_indices = np.where((y_true == 0) & (mean_preds_flat > 0.90))[0]
    idx_fp = fp_indices[0] if len(fp_indices) > 0 else None
    fn_indices = np.where((y_true == 1) & (mean_preds_flat < 0.10))[0]
    idx_fn = fn_indices[0] if len(fn_indices) > 0 else None

    explainer = lime_tabular.LimeTabularExplainer(
        data["X_train_numpy"], feature_names=data["feature_names"], 
        class_names=['Healthy', 'Diabetes'], mode='classification', verbose=False
    )

    analyze_single_case(idx_most_unc, data, model, explainer, "Case 1 - Most Uncertain", run_dir)
    if idx_fp: analyze_single_case(idx_fp, data, model, explainer, "Case 2 - Confident False Positive", run_dir)
    if idx_fn: analyze_single_case(idx_fn, data, model, explainer, "Case 3 - Confident False Negative", run_dir)

    print(f"\n=== HOÀN TẤT! KẾT QUẢ TẠI: {run_dir} ===")

if __name__ == "__main__":
    main()