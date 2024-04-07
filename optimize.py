# optimize.py

import numpy as np
import math
import random
from autograd import grad
from scipy.optimize import minimize
from collections import OrderedDict
from datetime import datetime

from hypergraph import Hypergraph
from draw import Drawer

k_PR = 0.30
k_PA = 0.16
k_PS = 0.36
k_PI = 0.18

def set_k(pr, pa, ps, pi):
    """设置objective function系数"""
    global k_PR, k_PA, k_PS, k_PI
    k_PR = pr
    k_PA = pa
    k_PS = ps
    k_PI = pi


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

def get_PR(edge, edge_points):
    """计算单个多边形的PR"""
    P = perimeter(edge, edge_points)
    A = area(edge, edge_points)
    C = 4*len(edge)*(math.tan(math.pi/len(edge)))
    return P ** 2 - C * A

def get_PA(edge, points):
    """计算单个多边形的PA"""
    single_PA = 0
    n = len(points)
    for i in range(n):
        single_PA += (1 - distance(points[edge[i]], points[edge[(i + 1)% n]])) ** 2
    return single_PA

def get_PS(e1, e2, c1, c2):
    """计算两个给定多边形的PS"""
    db = 0.1 # buffer distance
    ab = math.pi/18 # buffer angle
    e = 0

    intersec = list([x for x in e1 if x in e2])
    if(len(intersec) == 1):
        p = e1[intersec[0]]
    n1 = len(e1)
    n2 = len(e2)
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

def get_PI(e1, e2, p1, p2):
    """计算两个给定多边形的PI"""

    intersec = [x for x in e1 if x in e2]
    intersec_points = {key: value for key, value in p1.items() if key in intersec}
    if(len(intersec) < 3):
        return 0
    
    n0 = len(intersec)
    n1 = len(e1)
    n2 = len(e2)

    s1 = {}
    s2 = {}
    for i in range(len(intersec)):
        s1[i] = 0
        s2[i] = 0
    
    index = 0
    for i in range(len(e1)):
        if e1[i] == intersec[index]:
            index = (index + 1) % len(intersec) 
            s1[index] += distance(p1[e1[i]], p1[e1[(i + 1) % len(e1)]])
        else:
            s1[index] += distance(p1[e1[i]], p1[e1[(i + 1) % len(e1)]])
    
    index = 0
    for i in range(len(e2)):
        if e2[i] == intersec[index]:
            index = (index + 1) % len(intersec) 
            s2[index] += distance(p2[e2[i]], p2[e2[(i + 1) % len(e2)]])
        else:
            s2[index] += distance(p2[e2[i]], p2[e2[(i + 1) % len(e2)]])

    e = sum([(x - n1/n0)**2 for x in s1]) + sum([(x - n2/n0)**2 for x in s2]) + get_PR(intersec,intersec_points)

    return e

def objective_function(x):
    """目标函数"""

    E_PR = 0 
    E_PA = 0
    E_PS = 0
    E_PI = 0

    # x中保存的一维坐标赋值回坐标向量
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
        Graph.edges[i] = list(edges_points[i].keys())

    for i in range(len(Graph.edges)):
        edge = Graph.edges[i]
        edge_points = edges_points[i]

        # Polygon Regularity (PR) Energy
        E_PR += get_PR(edge, edge_points)

        # Polygon Area (PA) Energy
        E_PA += get_PA(edge, edge_points)

    for i in range(len(Graph.edges)):
        for j in range(i+1,len(Graph.edges)):
            if(len(Graph.edges[i]) != 1 and len(Graph.edges[j]) != 1):
                # Polygon Separation (PS) Energy
                E_PS += get_PS(edges_points[i],edges_points[j],centroid_point[i],centroid_point[j])
                # Polygon Intersection (PI) Energy
                E_PI += get_PI(Graph.edges[i],Graph.edges[j],edges_points[i],edges_points[j])
            else:
                # Polygon Separation (PS) Energy
                # TODO：case for monogon
                pass
    Graph.points = points
    return k_PR * E_PR + k_PA * E_PA + k_PS * E_PS + k_PI * E_PI

def swap_minimize(points):
    y_buffer = 99999
    for e in Graph.edges:
        for i in range(len(e)):
            for j in range(i+1,len(e)):
                points_buffer = points.copy()
                buffer = points_buffer[e[i]]
                points_buffer[e[i]] = points_buffer[e[j]]
                points_buffer[e[j]] = buffer
                res = minimize(objective_function, get_x(points_buffer), method='L-BFGS-B')
                y = objective_function(res.x)
                if y < y_buffer:
                    print(e[i], e[j], y)
                    y_buffer = y
                    record = [e[i], e[j]]

    print("minimize swap:",record[0], record[1], y_buffer)
    if y_buffer < objective_function(get_x(points)):
        print("accepted swap:",record[0], record[1], y_buffer)
        buffer = points[record[0]]
        points[record[0]] = points[record[1]]
        points[record[1]] = buffer
        return points
    print("not accepted swap:",record[0], record[1], y_buffer)
    return points

# gradient_function = grad(objective_function)

# evaluations = []
# def callback_function(x):
#     evaluations.append(objective_function(x))

# res = minimize(objective_function, x, jac=gradient_function, method='L-BFGS-B', callback=callback_function)

def get_x(points):
    x = []
    for point in points:
        x.append(points[point][0])
        x.append(points[point][1])
    x = np.array(x)
    return x

def get_points(x):
    points = {}
    for i in range(len(x)//2):
        points[Graph.v[i]] = [x[i*2], x[i*2+1]]
    return points

def getres(graph:Hypergraph):
    """获得优化结果:顶点坐标集合"""
    global Graph 
    Graph = graph

    x = get_x(Graph.points)

    record = {}
    time = {}
    set_k(0.10, 0.08, 0.36, 0.18)
    record[0] = objective_function(x)
    time[0] = datetime.now().strftime("%H:%M:%S")


    # set_k(0.10, 0.08, 0.36, 0.18)
    res = minimize(objective_function, x, method='L-BFGS-B')
    points = get_points(res.x)

    
    for i in range(10):
        
        record[i+1] = objective_function(res.x)
        time[i+1] = datetime.now().strftime("%H:%M:%S")

        improve_note = 0

        draw = Drawer(graph, f"records/v-3-0/{i}.png",'Hypergraph Visualization', False)
        print("----------------------",i)
        
        # set_k(0, 0, 0.36, 0.18)
        points = swap_minimize(points)
        graph.points = points.copy()

        x = get_x(points)
        # set_k(0.10, 0.08, 0.36, 0.18)
        res = minimize(objective_function, x, method='L-BFGS-B')
        points = get_points(res.x)

        # if(objective_function(res.x) < min):
        #     min = objective_function(res.x)
        #     x_buffer = res.x.copy()

        if improve_note == 0:
            break

    # graph.points = get_points(x_buffer).copy()
    # draw = Drawer(graph, f"records/v-2-1/opt.png",'Hypergraph Visualization', False)
    
    print()
    print(record)
    print(time)
    print()

    return points
