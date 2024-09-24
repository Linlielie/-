import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置中文字体
mpl.rc('font', family='Microsoft YaHei')

class LinearRegression:
    def __init__(self):
        self.theta = None

    def fit(self, input_datax, datay, learning_rate=0.01, lamda=0.01, maxstep=1e6, stoplearn_count=50):
        sample_num, character_num = input_datax.shape
        datax = np.c_[input_datax, np.ones(sample_num)]  # 添加截距项
        self.theta = np.zeros((character_num + 1, 1))  # 初始化theta
        last_better = 0
        last_theta = float('inf')  # 初始化损失为无穷大

        for step in range(int(maxstep)):
            predict = datax.dot(self.theta)
            jtheta = np.mean((predict - datay) ** 2) / 2
            gradient = lamda * self.theta + (datax.T.dot(predict - datay)) / sample_num
            self.theta -= learning_rate * gradient

            if jtheta < last_theta - 1e-8:
                last_theta = jtheta
                last_better = step
            elif step - last_better > stoplearn_count:
                break

    def predict(self, input_datax):
        sample_num = input_datax.shape[0]
        datax = np.c_[input_datax, np.ones(sample_num)]
        return datax.dot(self.theta)

# 加载数据
data = pd.read_csv("housing-data.csv", header=None, delim_whitespace=True)  # 使用delim_whitespace处理空格
datax = data.iloc[:, :-1].values.astype(float)
datay = data.iloc[:, -1].values.reshape(-1, 1).astype(float)


# 特征归一化
datax = (datax - datax.min(axis=0)) / (datax.max(axis=0) - datax.min(axis=0))

# 训练模型
model = LinearRegression()
model.fit(datax, datay, learning_rate=0.5, lamda=0.00003)

# 预测数据
predict_data = model.predict(datax)

# 绘制结果
plt.figure()
plt.scatter(range(len(datay)), datay, color='g', label="实际值", s=30)
plt.plot(range(len(predict_data)), predict_data, 'r-', lw=1.6, label="预测值")
plt.legend(loc='best')
plt.title('波士顿房价预测', fontsize=18)
plt.xlabel('案例 ID', fontsize=15)
plt.ylabel('房价', fontsize=15)
plt.grid(True)
plt.show()
