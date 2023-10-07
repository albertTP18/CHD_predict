'''
    Thuật toán Gradient Descent - thủ tục Constant Stepsize
    Bài toán hồi quy logistic
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time

# Hàm sigmoid
def sigmoid(S):
    return 1 / (1 + np.exp(-S))
    # a Nx1 array

# Hàm tính xác suất của x - giá trị y dự đoán
def y_predict(w, x):
    return sigmoid(x.dot(w))
    # a Nx1 array

# Hàm mất mát (sử dụng hệ số chính quy hóa)
def loss(w, x, y):
    z = y_predict(w, x)
    return np.mean(- y * np.log(z) - (1 - y) * np.log(1 - z) + lam / 2 * np.linalg.norm(w)**2)
    # a number

# Gradient của hàm mất mát (sử dụng hệ số chính quy hóa)
def grad_loss(w, x, y):
    z = y_predict(w, x)
    return np.dot(x.T, z - y) / x.shape[0] + lam * w
    # a dx1 array

# Gradient Descent (constStepsize)
def GD_cs(w, x, y, t, epsilon):
    i = 1
    loss_hist = [loss(w, x, y)]
    gradLoss_hist = [np.linalg.norm(grad_loss(w, x, y))]
    while True:
        w_new = w - t * grad_loss(w, x, y)
        loss_hist.append(loss(w_new, x, y))
        gradLoss_hist.append(np.linalg.norm(grad_loss(w_new, x, y)))
        if np.linalg.norm(grad_loss(w_new, x, y)) < epsilon:
            print("Số vòng lặp:", i)
            return w_new, loss_hist, gradLoss_hist
        else:
            w = w_new
            i = i + 1
            print(i, loss(w_new, x, y), np.linalg.norm(grad_loss(w_new, x, y)))

# Load data từ file csv
data = pd.read_csv('CHD_preprocessed.csv').values
N, d = data.shape
x = data[:, 0:d - 1]
y = data[:, d - 1].reshape(-1, 1)

# Thêm cột giá trị 1 vào dữ liệu x
x = np.hstack((np.ones((N, 1)), x))

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Bước khởi tạo cho thuật toán GD_bt
w = np.full(x.shape[1], 0).reshape(-1, 1)
t = 0.00001
epsilon = 0.05
lam = 0.01

# Chạy thuật toán GD_bt và tính thời gian thực thi
t1 = time.time()  # Lưu thời điểm trước khi chạy
w, loss_hist, gradLoss_hist = GD_cs(w, x_train, y_train, t, epsilon)
t2 = time.time()  # Lưu thời điểm sau khi chạy

print("Trọng số cần tìm:", w.T)
print("Thời gian thực thi:", t2 - t1, "seconds")

# Vẽ giá trị hàm loss
plt.plot(loss_hist)
plt.xlabel('Vòng lặp', fontsize=13)
plt.ylabel('Hàm mất mát', fontsize=13)
plt.tick_params(axis='both', which='major', labelsize=13)

plt.show()

# Vẽ gradient hàm loss
plt.plot(gradLoss_hist)
plt.xlabel('Vòng lặp', fontsize=13)
plt.ylabel('Chuẩn của gradient hàm mất mát', fontsize=13)
plt.tick_params(axis='both', which='major', labelsize=13)

plt.show()

# Tính giá trị y dự đoán của tập kiểm tra
y_predict_test = y_predict(w, x_test)

# Chuyển thành dạng nhị phân sử dụng ngưỡng 0.5
threshold = 0.5
y_predict_test = np.where(y_predict_test >= threshold, 1, 0)

# Đánh giá accuracy
accuracy = accuracy_score(y_test, y_predict_test)
print("Độ chính xác", accuracy)

