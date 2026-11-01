import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Tải dữ liệu
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Cấu hình tốt nhất lấy từ Optuna (Trial 3 trong file log)
best_params = {
    'n_estimators': 250,
    'max_depth': 12,
    'min_samples_split': 2,
    'min_samples_leaf': 1
}

# 3. Huấn luyện mô hình tối ưu
rf_optimized = RandomForestClassifier(**best_params, random_state=42)
rf_optimized.fit(X_train, y_train)

# 4. Đánh giá trên tập Test
y_pred = rf_optimized.predict(X_test)
optimized_accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy sau khi tối ưu (Optuna): {optimized_accuracy:.4f}")
# Baseline: 0.9649 -> Optimized: Có thể đạt 0.9736
