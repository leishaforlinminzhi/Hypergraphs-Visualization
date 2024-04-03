# optimize.py

import numpy as np
import math
from autograd import grad
from scipy.optimize import minimize

from hypergraph import Hypergraph

Graph = Hypergraph([],[])

class Optimizer:
    def __init__(self, graph:Hypergraph):
        self.res = getres(graph)


def distance(point1, point2):
    """计算两点之间的距离"""
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def perimeter(points):
    """计算图形的周长"""
    total_distance = 0
    n = len(points)
    for i in range(n):
        total_distance += distance(points[i], points[(i + 1) % n])
    return total_distance

def area(points):
    """计算图形的面积"""
    n = len(points)
    total_area = 0
    for i in range(n):
        j = (i + 1) % n
        total_area += (points[i][0] * points[j][1] - points[j][0] * points[i][1])
    return abs(total_area) / 2

def objective_function(x):
    # 将x中保存的一维坐标赋值回坐标向量
    points = {}
    for i in range(len(x)//2):
        points[Graph.v[i]] = [x[i*2], x[i*2+1]]
    print("points:",points)

    # Polygon Regularity (PR) Energy:
    E_PR = 0 
    for edge in Graph.edges:
        print("edge:",edge)
        p = {key: value for key, value in points.items() if key in edge}
        print(p)
        P = perimeter(points)
        A = area(points)
        C = 4*len(points)/(math.tan(math.pi/len(points)))
        E_PR += P^2 - C * A
    print("--------------------")
    return E_PR 

gradient_function = grad(objective_function)

evaluations = []
def callback_function(x):
    evaluations.append(objective_function(x))

def getres(graph:Hypergraph):
    global Graph 
    Graph = graph
    x = []
    for point in Graph.points:
        x.append(Graph.points[point][0])
        x.append(Graph.points[point][1])
    
    x = np.array(x)
    res = minimize(objective_function, x, jac=gradient_function, method='L-BFGS-B', callback=callback_function)

    return res
