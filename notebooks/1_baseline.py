import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Tải dữ liệu Breast Cancer
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 2. Chia tập Train/Test (80% huấn luyện, 20% kiểm thử)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Khởi tạo Baseline: Mô hình Random Forest với siêu tham số mặc định
rf_baseline = RandomForestClassifier(random_state=42)
rf_baseline.fit(X_train, y_train)

# 4. Đánh giá Baseline trên tập Test
y_pred = rf_baseline.predict(X_test)
baseline_accuracy = accuracy_score(y_test, y_pred)

print(f"Baseline Accuracy (Cấu hình mặc định): {baseline_accuracy:.4f}")
# Kết quả thực tế khi chạy ở seed 42 là khoảng 0.9649
