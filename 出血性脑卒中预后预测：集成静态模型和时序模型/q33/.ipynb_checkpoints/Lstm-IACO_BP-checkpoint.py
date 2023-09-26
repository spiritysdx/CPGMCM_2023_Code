import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from aco import ACO  # 蚁群优化算法模块
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# 加载数据
data_with_time = pd.read_csv('data_with_time.csv')  # 带时间属性的数据
data_without_time = pd.read_csv('data_without_time.csv')  # 不带时间属性的数据

# 划分训练集和测试集
x_train_with_time, x_test_with_time, y_train, y_test = train_test_split(data_with_time.drop(['label'], axis=1), data_with_time['label'], test_size=0.2, random_state=42)
x_train_without_time, x_test_without_time, _, _ = train_test_split(data_without_time, data_without_time, test_size=0.2, random_state=42)

# 定义LSTM模型
def create_lstm_model(n_timesteps, n_features, n_classes):
    model = Sequential()
    model.add(LSTM(64, input_shape=(n_timesteps, n_features), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(32))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    return model

n_timesteps, n_features = x_train_with_time.shape[1], x_train_with_time.shape[2]
lstm_model = create_lstm_model(n_timesteps, n_features, n_classes)

# 定义ACO-BP神经网络模型
class ACO_BPNN:
    def __init__(self, max_iter, n_ants, alpha, beta, rho, q0, hidden_units):
        self.max_iter = max_iter
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.hidden_units = hidden_units
    
    # 计算预测准确率
    def accuracy(self, y_true, y_pred):
        return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
    
    # 训练模型
    def fit(self, x_train, y_train, x_test, y_test):
        n_features, n_classes = x_train.shape[1], y_train.shape[1]
        lstm_model = create_lstm_model(1, n_features, self.hidden_units)
        aco = ACO(self.max_iter, self.n_ants, self.alpha, self.beta, self.rho, self.q0)
        w = aco.search(lstm_model.count_params(), n_classes, self.accuracy, x_train, y_train, x_test, y_test)
        lstm_model.set_weights(w)
        return lstm_model

# 训练LSTM模型
lstm_model.fit(x_train_with_time, y_train, epochs=10)

# 训练ACO-BP神经网络模型
max_iter = 50  # 蚁群优化算法迭代次数
n_ants = 10  # 蚂蚁数量
alpha = 1.0  # 信息素启发因子
beta = 1.0  # 启发函数因子
rho = 0.5  # 信息素挥发因子
q0 = 0.7  # 蚁群中蚂蚁选择概率
hidden_units = 16
aco_bpnn = ACO_BPNN(max_iter, n_ants, alpha, beta, rho, q0, hidden_units)
aco_lstm_model = aco_bpnn.fit(x_train_without_time, y_train, x_test_without_time, y_test)

# 构建投票融合模型
voting_model = VotingClassifier(estimators=[('lstm', lstm_model), ('aco_bpnn', aco_lstm_model)], voting='soft')
voting_model.fit(x_train_with_time, y_train)

# 在测试集上评估模型性能
y_pred = voting_model.predict(x_test_with_time)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f%%' % (accuracy * 100))
