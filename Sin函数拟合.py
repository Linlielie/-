import numpy as np
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
        loss_history = []  # 用于记录损失

        for step in range(int(maxstep)):
            predict = datax.dot(self.theta)
            jtheta = np.mean((predict - datay) ** 2) / 2
            loss_history.append(jtheta)  # 记录损失
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

# 生成数据
x = np.linspace(-np.pi, np.pi, 8000).reshape(-1, 1)
y = np.sin(x)

# 多项式特征生成
datax = np.ones((x.shape[0], 6))
for i in range(1, 6):
    datax[:, i] = x[:, 0] ** i

# 标准化
mean = np.mean(datax, axis=0)
std = np.std(datax, axis=0)
std[std == 0] = 1e-9  # 避免除以零

datax = (datax - mean) / std

# 训练模型
model = LinearRegression()
model.fit(datax, y, learning_rate=0.3, lamda=0.00005)

# 预测
predict_y = model.predict(datax)

# 绘制结果
plt.figure()
plt.plot(x, y, "r-", lw=1.6, label='sin(x)')
plt.plot(x, predict_y, 'g-', lw=1.6, label='预测曲线')
plt.legend(loc='best')
plt.grid(True)
plt.title('线性回归预测与实际')
plt.xlabel('x 轴')
plt.ylabel('y 轴')
plt.show()
