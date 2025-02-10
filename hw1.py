import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = {
    '密度': [0.697, 0.774, 0.634, 0.608, 0.556, 0.403, 0.481, 0.437, 
           0.666, 0.243, 0.245, 0.343, 0.639, 0.657, 0.360, 0.593, 0.719],
    '含糖率': [0.460, 0.376, 0.264, 0.318, 0.215, 0.237, 0.149, 0.211, 
             0.091, 0.267, 0.057, 0.099, 0.161, 0.198, 0.370, 0.042, 0.103],
    '好瓜': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}


df = pd.DataFrame(data)


X = df[['密度', '含糖率']]  
y = df['好瓜']  


model = LogisticRegression()
model.fit(X, y)


coefficients = model.coef_
intercept = model.intercept_

print(f"密度的系数: {coefficients[0][0]}")
print(f"含糖率的系数: {coefficients[0][1]}")
print(f"截距: {intercept[0]}")


y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)

print("预测的类别：", y_pred)
print("实际的类别：", y.values)
print(f"模型准确率: {accuracy * 100:.2f}%")
