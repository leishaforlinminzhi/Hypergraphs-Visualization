# optimize.py

import numpy as np
import math
from autograd import grad
from scipy.optimize import minimize
from collections import OrderedDict

from hypergraph import Hypergraph

Graph = Hypergraph([],[])

class Optimizer:
    def __init__(self, graph:Hypergraph):
        self.res = getres(graph)

def centroid(points):
    """多边形顶点按照质心极坐标重新排序"""
    centroid_x = 0
    centroid_y = 0

    for point in points.values():
        centroid_x += point[0]
        centroid_y += point[1]

    centroid_x /= len(points)
    centroid_y /= len(points)

    angle_dict = {}
    for key, point in points.items():
        relative_x = point[0] - centroid_x
        relative_y = point[1] - centroid_y
        angle_dict[key] = math.atan2(relative_y, relative_x)
    
    sorted_points = sorted(points.items(), key=lambda item: angle_dict[item[0]])

    sorted_points = {key: value for key, value in sorted_points}

    return sorted_points

def distance(point1, point2):
    """计算两点之间的距离"""
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def perimeter(edge, points):
    """计算图形的周长"""
    total_distance = 0
    n = len(points)
    for i in range(n):
        total_distance += distance(points[edge[i]], points[edge[(i + 1)% n]])
    return total_distance

def area(edge, points):
    """计算图形的面积"""
    n = len(points)
    total_area = 0
    for i in range(n):
        j = (i + 1) % n
        total_area += (points[edge[i]][0] * points[edge[j]][1] - points[edge[j]][0] * points[edge[i]][1])
    return abs(total_area) / 2

def objective_function(x):
    # 将x中保存的一维坐标赋值回坐标向量
    points = {}
    for i in range(len(x)//2):
        points[Graph.v[i]] = [x[i*2], x[i*2+1]]
    
    edges_points = {}
    # 将每条边对应的点通过质心极坐标排序
    for i in range(len(Graph.edges)):
        edge = Graph.edges[i]
        edges_points[i] = {key: value for key, value in points.items() if key in edge}
        # print("before:",edges_points[i])
        edges_points[i] = centroid(edges_points[i])
        edge = list(edges_points[i].keys())
        # print("after:",edges_points[i])
    
    # Polygon Regularity (PR) Energy:
    E_PR = 0 
    for i in range(len(Graph.edges)):
        edge = Graph.edges[i]
        edge_points = edges_points[i]
        P = perimeter(edge, edge_points)
        A = area(edge, edge_points)
        C = 4*len(edge)*(math.tan(math.pi/len(edge)))
        E_PR += P * P - C * A
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
    # res = minimize(objective_function, x, jac=gradient_function, method='L-BFGS-B', callback=callback_function)
    res = minimize(objective_function, x, method='L-BFGS-B')
    print(res.x)
    points = {}
    for i in range(len(x)//2):
        points[Graph.v[i]] = [res.x[i*2], res.x[i*2+1]]

    return points
