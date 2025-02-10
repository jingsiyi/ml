# 导入必要的库
import numpy as np
from tensorflow import Se

# 生成简单的时间序列数据
def generate_time_series(n_samples, n_steps):
    X = np.random.rand(n_samples, n_steps)
    y = X.sum(axis=1)  # 目标是时间步的总和
    return X[..., np.newaxis], y  # 添加一个维度以匹配 LSTM 的输入格式

# 超参数
n_samples = 1000
n_steps = 10
batch_size = 32
epochs = 20

# 数据准备
X, y = generate_time_series(n_samples, n_steps)
train_size = int(0.8 * n_samples)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 构建 LSTM 模型
model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_steps, 1)),  # 50个单元的LSTM层
    Dense(1)  # 输出层
])

# 编译模型
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 训练模型
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=epochs,
                    batch_size=batch_size)

# 模型评估
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# 使用模型预测
predictions = model.predict(X_test, verbose=0)

# 可视化结果
import matplotlib.pyplot as plt
plt.plot(y_test, label='True Values')
plt.plot(predictions, label='Predictions')
plt.legend()
plt.title('LSTM Time Series Prediction')
plt.show()
