def f(x):
    return -0.00108702385*x**3-0.0163570597*x**2+2.92372561*x+80.6226732

y0=86.396

from scipy.optimize import fsolve
import numpy as np


# 求解方程 f(x) = y
def solve_equation(y):
    equation = lambda x: f(x) - y
    x_initial_guess = 0  # 初始猜测值
    x_solution = fsolve(equation, x_initial_guess)
    return x_solution[0]

# 给定 y 值
y = y0+6

# 求解 x
x = solve_equation(y)
print("对应的 x 值：", x)
