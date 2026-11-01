# Tối ưu hóa siêu tham số cho mô hình Random Forest

## 1. Đặt vấn đề (Problem Statement)
Việc tinh chỉnh siêu tham số (Hyperparameter tuning) cho các mô hình học máy thường tốn nhiều thời gian nếu làm thủ công. Dự án này nhằm mục tiêu tự động hóa quá trình tìm kiếm cấu hình tối ưu cho mô hình Random Forest trên tập dữ liệu Breast Cancer Wisconsin bằng thuật toán tối ưu.

## 2. Mô hình hóa bài toán (Optimization Formulation)
*   **Biến quyết định (Decision variables $x$):** Vector chứa các siêu tham số cần tối ưu của thuật toán Random Forest: $x = [n\_estimators, max\_depth, min\_samples\_split, min\_samples\_leaf]$.
*   **Hàm mục tiêu (Objective $f(x)$):** Cực đại hóa độ chính xác (Accuracy) trên tập kiểm thử (Test set).
    $$\max f(x) = \text{Accuracy}(x)$$
*   **Ràng buộc (Constraints / Budget):**
    *   $50 \le n\_estimators \le 300$
    *   $2 \le max\_depth \le 20$
    *   Số lần chạy tối đa (Budget constraint): Thuật toán chỉ được chạy tối đa 50 trials.

## 3. Cấu trúc thư mục (Repository Structure)
*   `notebooks/`: Chứa mã nguồn thực nghiệm.
*   `results/`: Chứa file log lịch sử chạy và biểu đồ minh họa.
*   `report/`: Chứa báo cáo môn học.
*   `data/`: Chứa thông tin về nguồn dataset.
