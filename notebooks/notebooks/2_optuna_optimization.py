import pandas as pd
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# 1. Tải dữ liệu và chia tập Train/Test
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Định nghĩa hàm mục tiêu (Objective function) cho Optuna
def objective(trial):
    # Xác định không gian tìm kiếm (Decision variables)
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 2, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    
    # Khởi tạo mô hình
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    # Đánh giá bằng Cross Validation trên tập Train (KHÔNG dùng tập Test để tuning)
    score = cross_val_score(rf, X_train, y_train, n_jobs=-1, cv=3)
    return score.mean()

# 3. Chạy tối ưu hóa với budget = 50 trials
print("Bắt đầu quá trình tối ưu hóa...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Cấu hình tốt nhất:", study.best_params)

# 4. BẮT BUỘC: Code lưu lịch sử chạy ra file CSV để báo cáo
df_trials = study.trials_dataframe()
# df_trials.to_csv('../results/optuna_trials.csv', index=False)
print("Đã lưu lịch sử chạy 50 trials ra file CSV.")
