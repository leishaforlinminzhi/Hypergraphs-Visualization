# optimize.py

import numpy as np
import math
import random
import json
from autograd import grad
from scipy.optimize import minimize
from collections import OrderedDict
from datetime import datetime

from hypergraph import Hypergraph
from draw import Drawer
from SwapRecord import SwapRecord, Stack

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
subGraph = Hypergraph([],[])
swapNote = []
edgeNote = []

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

def objective_function_sub(x):
    """子图目标函数"""

    E_PR = 0 
    E_PA = 0
    E_PS = 0
    E_PI = 0

    # x中保存的一维坐标赋值回坐标向量
    points = {}
    for i in range(len(x)//2):
        points[subGraph.v[i]] = [x[i*2], x[i*2+1]]
    
    edges_points = {}
    centroid_point = {}
    # 将每条边对应的点通过质心极坐标排序
    for i in range(len(subGraph.edges)):
        edge = subGraph.edges[i]
        edges_points[i] = {key: value for key, value in points.items() if key in edge}
        edges_points[i],centroid_point[i] = centroid(edges_points[i])
        subGraph.edges[i] = list(edges_points[i].keys())

    for i in range(len(subGraph.edges)):
        edge = subGraph.edges[i]
        edge_points = edges_points[i]

        # Polygon Regularity (PR) Energy
        E_PR += get_PR(edge, edge_points)

        # Polygon Area (PA) Energy
        E_PA += get_PA(edge, edge_points)

    for i in range(len(subGraph.edges)):
        for j in range(i+1,len(subGraph.edges)):
            if(len(subGraph.edges[i]) != 1 and len(subGraph.edges[j]) != 1):
                # Polygon Separation (PS) Energy
                E_PS += get_PS(edges_points[i],edges_points[j],centroid_point[i],centroid_point[j])
                # Polygon Intersection (PI) Energy
                E_PI += get_PI(subGraph.edges[i],subGraph.edges[j],edges_points[i],edges_points[j])
            else:
                # Polygon Separation (PS) Energy
                # TODO：case for monogon
                pass
    subGraph.points = points
    return k_PR * E_PR + k_PA * E_PA + k_PS * E_PS + k_PI * E_PI


def get_subgraph(p1, p2):
    graph = Hypergraph([],[])
    graph.edges = []
    graph.v = set()

    for e in Graph.eofv[p1]:
        if Graph.edges[e] not in graph.edges:
            graph.edges.append(Graph.edges[e])
            graph.v.update(Graph.edges[e])
    for e in Graph.eofv[p2]:
        if Graph.edges[e] not in graph.edges:
            graph.edges.append(Graph.edges[e])
            graph.v.update(Graph.edges[e])

    vs = list(graph.v)

    for v in vs:
        for e in Graph.eofv[v]:
            if Graph.edges[e] not in graph.edges:
                graph.edges.append(Graph.edges[e])
                graph.v.update(Graph.edges[e])

    graph.v = list(graph.v)

    graph.points = {key: value for key, value in Graph.points.items() if key in graph.v}

    return graph

def swap_minimize(points):
    global swapNote
    global edgeNote

    swapStack = Stack()

    for k in range(len(Graph.edges)):
        e = Graph.edges[k]
        for i in range(len(e)):
            for j in range(i+1,len(e)):
                # 剪枝
                if(Graph.d[e[i]] == 1 and Graph.d[e[j]] == 1) or len(e) < 3 or swapNote[e[i]][e[j]] == 1:
                    continue
                
                # 取子图
                global subGraph
                subGraph = get_subgraph(e[i],e[j])
                
                y_buffer = objective_function_sub(get_x(subGraph.points))

                # 剪枝 
                if y_buffer < 0.001:
                    continue
                
                buffer = subGraph.points[e[i]]
                subGraph.points[e[i]] = subGraph.points[e[j]]
                subGraph.points[e[j]] = buffer

                res = minimize(objective_function_sub, get_x(subGraph.points), method='L-BFGS-B')

                y = objective_function_sub(res.x)

                print(Graph.edges[k], e[i], e[j], ((y_buffer - y) * edgeNote[k]))

                record = SwapRecord()
                record.pair = [e[i], e[j]]
                record.edge = k
                record.points = subGraph.points.copy()
                record.energy = (y_buffer - y) * edgeNote[k]

                if((y_buffer - y) * edgeNote[k]) > 0:
                    swapStack.push(record)
    
    y_points = objective_function(get_x(points))
    points_buffer = points.copy()

    while not (swapStack.is_empty()):
        points = points_buffer.copy()

        record = swapStack.pop()

        for p in record.points:
            points[p] = record.points[p]

        res = minimize(objective_function, get_x(points), method='L-BFGS-B')

        print(record.pair, objective_function(res.x), y_points)
        if(objective_function(res.x) < y_points):
            swapNote[record.pair[0]][record.pair[1]] = 1
            swapNote[record.pair[1]][record.pair[0]] = 1
            # 边内有点对交换过的边优先级降低
            edgeNote[record.edge] *= 0.8
            print("accepted swap:",record.pair[0], record.pair[1])
            return get_points(res.x)
        else:
            # 边内点对交换失败的边优先级降低
            edgeNote[record.edge] *= 0.9
    print("not accepted swap")
    return points

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
    global Graph, swapNote, edgeNote
    Graph = graph
    swapNote = [[0 for j in range(max(Graph.v)+2)] for i in range(max(Graph.v)+2)]
    edgeNote = [1 for i in range(len(Graph.edges)+2)]

    x = get_x(Graph.points)

    record = {}
    time = {}
    set_k(0.08, 0.06, 0.36, 0.18)
    record[0] = objective_function(x)
    time[0] = datetime.now().strftime("%H:%M:%S")

    res = minimize(objective_function, x, method='L-BFGS-B')
    points = get_points(res.x)
    buffer = objective_function(res.x)
    
    for i in range(50):
        
        record[i+1] = objective_function(res.x)
        time[i+1] = datetime.now().strftime("%H:%M:%S")


        draw = Drawer(graph, f"records/v-4-1/{i}.png",'Hypergraph Visualization', False)
        print("----------------------",i)
        
        points = swap_minimize(points)
        graph.points = points.copy()

        x = get_x(points)
        res = minimize(objective_function, x, method='L-BFGS-B')
        points = get_points(res.x)

        y = objective_function(res.x)
        if(y < 0.01):
            break

    print()
    print(record)
    print(time)
    print()

    filename = 'records/v-4-1/record.txt'
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            string1 = (str)(json.dumps(record))
            string2 = (str)(json.dumps(time))
            file.write(string1+string2)
    except Exception as e:
        print("写入文件时出错:", str(e))

    return points
