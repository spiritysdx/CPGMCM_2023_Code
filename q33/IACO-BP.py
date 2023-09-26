'''
利用一个优化过的蚁群算法来优化BP神经网络，解决第三问a小问
'''
import re
import math
import warnings
import tqdm
import copy
import matplotlib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from scipy.optimize import fsolve
from sympy import symbols, Eq, solve
from sklearn.preprocessing import MinMaxScaler
from npp import newff, train, sim
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
warnings.filterwarnings("ignore")


plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 



# 定义神经网络类
class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        # 初始化权重矩阵
        self.weights_ih = np.random.randn(self.hidden_nodes, self.input_nodes)
        self.weights_ho = np.random.randn(self.output_nodes, self.hidden_nodes)
        
        # 初始化偏置项
        self.bias_h = np.random.randn(self.hidden_nodes, 1)
        self.bias_o = np.random.randn(self.output_nodes, 1)
    
    def forward(self, inputs):
        # 前向传播计算输出
        inputs = np.array(inputs, ndmin=2).T
        hidden_inputs = np.dot(self.weights_ih, inputs) + self.bias_h
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.weights_ho, hidden_outputs) + self.bias_o
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
    
    def backward(self, inputs, targets, learning_rate):
        # 反向传播更新权重矩阵和偏置项
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T
        hidden_inputs = np.dot(self.weights_ih, inputs) + self.bias_h
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.weights_ho, hidden_outputs) + self.bias_o
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.weights_ho.T, output_errors)
        
        self.weights_ho += learning_rate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), hidden_outputs.T)
        self.bias_o += learning_rate * output_errors
        
        self.weights_ih += learning_rate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), inputs.T)
        self.bias_h += learning_rate * hidden_errors
        
    @staticmethod
    def activation_function(x):
        return 1.0 / (1.0 + np.exp(-x))

# 定义蚁群算法类
class AntColonyOptimization:
    def __init__(self, population_size, num_cities, num_iterations, alpha, beta, rho, q):
        self.population_size = population_size
        self.num_cities = num_cities
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        
        self.distances = np.zeros((self.num_cities, self.num_cities))
        self.pheromone_trails = np.ones((self.num_cities, self.num_cities))
        
        self.best_solution = float('inf')
        self.best_path = []
    
    def optimize(self):
        for _ in range(self.num_iterations):
            self.ant_population = self.generate_ants()
            
            for ant in self.ant_population:
                ant.construct_solution()
                ant.update_pheromone_trails()
                
                if ant.path_length < self.best_solution:
                    self.best_solution = ant.path_length
                    self.best_path = ant.path
    
            self.update_pheromone_trails()
    
    def generate_ants(self):
        ants = []
        
        for _ in range(self.population_size):
            ant = Ant(self.num_cities, self.distances, self.pheromone_trails, self.alpha, self.beta)
            ants.append(ant)
            
        return ants
    
    def update_pheromone_trails(self):
        self.pheromone_trails *= (1.0 - self.rho)
        
        for ant in self.ant_population:
            delta_pheromone = self.q / ant.path_length
            
            for i in range(self.num_cities - 1):
                city_i = ant.path[i]
                city_j = ant.path[i + 1]
                self.pheromone_trails[city_i][city_j] += delta_pheromone
                self.pheromone_trails[city_j][city_i] += delta_pheromone

# 定义蚂蚁类
class Ant:
    def __init__(self, num_cities, distances, pheromone_trails, alpha, beta):
        self.num_cities = num_cities
        self.distances = distances
        self.pheromone_trails = pheromone_trails
        self.alpha = alpha
        self.beta = beta
        
        self.path = []
        self.path_length = 0.0
        self.unvisited_cities = [i for i in range(num_cities)]
        self.current_city = np.random.randint(0, num_cities)
        
        self.path.append(self.current_city)
        self.unvisited_cities.remove(self.current_city)
    
    def construct_solution(self):
        while len(self.unvisited_cities) > 0:
            next_city = self.select_next_city()
            self.path.append(next_city)
            self.unvisited_cities.remove(next_city)
            
            self.path_length += self.distances[self.current_city][next_city]
            self.current_city = next_city
    
    def select_next_city(self):
        visibility = 1.0 / self.distances[self.current_city]
        pheromone = self.pheromone_trails[self.current_city]
        
        probabilities = np.power(pheromone, self.alpha) * np.power(visibility, self.beta)
        probabilities /= np.sum(probabilities)
        
        next_city = np.random.choice(range(self.num_cities), p=probabilities)
        
        return next_city
    
    def update_pheromone_trails(self):
        delta_pheromone = 1.0 / self.path_length
        
        for i in range(self.num_cities - 1):
            city_i = self.path[i]
            city_j = self.path[i + 1]
            self.pheromone_trails[city_i][city_j] += delta_pheromone
            self.pheromone_trails[city_j][city_i] += delta_pheromone
            
# 调用3a,3b数据集进行测试
data = pd.read_excel('3a数据提取结果.xlsx')[:100]#只取前100行训练
X = data[2:]
y = data['90天mRS']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

input_nodes = X_train.shape[1]
hidden_nodes = 200
output_nodes = len(set(y_train))

neural_network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)

distances = np.zeros((output_nodes, output_nodes))
for i in range(output_nodes):
    for j in range(output_nodes):
        if i != j:
            distances[i][j] = np.linalg.norm(neural_network.weights_ih[i] - neural_network.weights_ho[j])

aco = AntColonyOptimization(population_size=10, num_cities=output_nodes, num_iterations=20, alpha=1.0, beta=2.0, rho=0.5, q=1.0)

# 优化神经网络权重矩阵
aco.optimize()

best_path = aco.best_path

weights_ih = np.zeros_like(neural_network.weights_ih)
weights_ho = np.zeros_like(neural_network.weights_ho)

for i in range(output_nodes):
    weights_ih[i] = neural_network.weights_ih[best_path[i]]

for i in range(output_nodes):
    weights_ho[i] = neural_network.weights_ho[best_path[i]]

neural_network.weights_ih = weights_ih
neural_network.weights_ho = weights_ho

predictions = []
for sample in X_test:
    prediction = np.argmax(neural_network.forward(sample))
    predictions.append(prediction)

accuracy = np.sum(predictions == y_test) / len(y_test)
print("测试集准确率：", accuracy)
