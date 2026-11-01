import optuna
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

def main():
    print("1. Đang tải tập dữ liệu Breast Cancer...")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # Chia dữ liệu: 80% train, 20% test. Dùng random_state để đảm bảo tính tái lập (Reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("2. Đang thiết lập hàm mục tiêu (Objective)...")
    def objective(trial):
        # Định nghĩa biến quyết định (Decision Variables) và miền giá trị (Search Space)
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 2, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
        
        # Khởi tạo mô hình với các tham số từ Optuna
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        
        # Đánh giá bằng Cross Validation (3-fold) trên tập TRAIN để tránh Data Leakage
        score = cross_val_score(rf, X_train, y_train, cv=3).mean()
        return score

    print("3. Bắt đầu quá trình Tối ưu hóa với Optuna (Budget: 50 trials)...")
    # Tạo study với mục tiêu tối đa hóa (maximize) accuracy
    study = optuna.create_study(direction='maximize', study_name="RandomForest_Optimization")
    study.optimize(objective, n_trials=50)

    print("\n" + "="*50)
    print("4. HOÀN TẤT TỐI ƯU HÓA")
    print("="*50)
    print(f"Độ chính xác tốt nhất (Validation Accuracy): {study.best_value:.4f}")
    print("Cấu hình siêu tham số (Hyperparameters) tốt nhất:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")

    print("\n5. Đang xuất file log kết quả và vẽ đồ thị...")
    # Xuất lịch sử ra file CSV để báo cáo GitHub
    df_trials = study.trials_dataframe()
    df_trials.to_csv('results/optuna_history.csv', index=False)
    print(" -> Đã lưu lịch sử chạy vào file 'optuna_history.csv'")

    # Vẽ đồ thị hội tụ (Optimization History)
    try:
        from optuna.visualization.matplotlib import plot_optimization_history
        plot_optimization_history(study)
        plt.title('Biểu đồ Lịch sử Hội tụ - Optuna')
        plt.tight_layout()
        plt.savefig("figures/optimization_history.png")
        plt.show()
    except ImportError:
        print(" -> (Cần cài đặt matplotlib để xem đồ thị trực tiếp)")

if __name__ == "__main__":
    main()