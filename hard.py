import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

psd = pd.read_csv("data_9.csv")

# 湿度序列
humidity = psd['RHU'].values.tolist()
# 降水序列
precipitation = psd['PRE_1h'].values.tolist()

# 将序列转换为二维数组
X = np.array(humidity).reshape(-1, 1)
y = np.array(precipitation)

# 创建随机森林回归模型
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 拟合模型
model.fit(X, y)

pds = pd.read_csv("data_10.csv")
# 预测降水序列
humidity_test = pds['RHU'].values.tolist()
X_test = np.array(humidity_test).reshape(-1, 1)
predicted_precipitation = model.predict(X_test)
predicted_precipitation = np.maximum(predicted_precipitation, 0)  # 将预测值限制为非负数

roof = pd.DataFrame()
roof['观测站点号'] = pds['Station_Id_C']
roof['预报时间'] = pds['Year'].astype(str).str.zfill(4) + pds['Mon'].astype(str).str.zfill(2) + pds['Day'].astype(str).str.zfill(2) + pds['Hour'].astype(str).str.zfill(2)
roof['实际1h降雨量'] = predicted_precipitation

# 添加实际3h降雨量列
roof['实际3h降雨量'] = predicted_precipitation[2:].tolist() + [0, 0]
# 添加实际6h降雨量列
roof['实际6h降雨量'] = predicted_precipitation[5:].tolist() + [0, 0, 0, 0, 0]

roof.to_excel("data_10_.xlsx", index=False)
# print(predicted_precipitation)