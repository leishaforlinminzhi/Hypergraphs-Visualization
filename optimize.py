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

    return sorted_points, [centroid_x,centroid_y]

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

def get_PA(edge, points):
    """计算单个多边形的PA"""
    single_PA = 0
    n = len(points)
    for i in range(n):
        single_PA += (1 - distance(points[edge[i]], points[edge[(i + 1)% n]])) ** 2
    return single_PA

def get_PS(intersec, p, n1, n2, c1, c2):
    db = 0.05 # buffer distance
    ab = math.pi/12 # buffer angle
    e = 0
    if len(intersec) == 0:
        e = distance(c1, c2) - (1/(math.sin(math.pi/n1)*2) + 
                                1/(math.sin(math.pi/n2)*2) + db)
    elif len(intersec) == 1:
        a0 = math.pi * ((n1 - 2) / (2 * n1) + (n2 - 2) / (2 * n2)) + ab

        AB = (p[0] - c1[0], p[1] - c1[1])
        BC = (p[0] - c2[0], p[1] - c2[1])
        dot_product = AB[0] * BC[0] + AB[1] * BC[1]
        cosine_theta = dot_product / (distance(p,c1) * distance(p,c2))
        a = math.acos(cosine_theta)

        e = a - a0
    elif len(intersec) == 2:
        e = distance(c1, c2) - (1/(math.tan(math.pi/n1)) + 
                                1/(math.tan(math.pi/n2)))/2
    if e <= 0:
        return e ** 2
    else:
        return 0
        
def objective_function(x):
    # 将x中保存的一维坐标赋值回坐标向量
    points = {}
    for i in range(len(x)//2):
        points[Graph.v[i]] = [x[i*2], x[i*2+1]]
    
    edges_points = {}
    centroid_point = {}
    # 将每条边对应的点通过质心极坐标排序
    for i in range(len(Graph.edges)):
        edge = Graph.edges[i]
        edges_points[i] = {key: value for key, value in points.items() if key in edge}
        edges_points[i],centroid_point[i] = centroid(edges_points[i])
        edge = list(edges_points[i].keys())
    
    k_PR = 0.30
    k_PA = 0.16
    k_PS = 0.36
    k_PI = 0.18

    E_PR = 0 
    E_PA = 0
    E_PS = 0
    E_PI = 0

    for i in range(len(Graph.edges)):
        edge = Graph.edges[i]
        edge_points = edges_points[i]

        # Polygon Regularity (PR) Energy
        P = perimeter(edge, edge_points)
        A = area(edge, edge_points)
        C = 4*len(edge)*(math.tan(math.pi/len(edge)))
        E_PR += P ** 2 - C * A

        # Polygon Area (PA) Energy
        E_PA += get_PA(edge, edge_points)

    # Polygon Separation (PS) Energy
    # TODO：case for monogon
    intersec_point = None
    for i in range(len(Graph.edges)):
        for j in range(i+1,len(Graph.edges)):
            seti = set(Graph.edges[i])
            setj = set(Graph.edges[j])
            intersec = list(seti.intersection(setj))
            if(len(intersec) == 1):
                intersec_point = points[intersec[0]]
            E_PS += get_PS(intersec,intersec_point,len(Graph.edges[i]),len(Graph.edges[j]),centroid_point[i],centroid_point[j])


    return k_PR * E_PR + k_PA * E_PA + k_PS * E_PS +k_PI * E_PI

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
