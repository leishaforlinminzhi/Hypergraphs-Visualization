# visualize.py

import argparse
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import minimize


from hypergraph import Hypergraph
from draw import Drawer
from optimize import Optimizer


def main(args):
    graph = Hypergraph([],[])
    graph.dataloader(args.filename)

    # 为每个顶点生成随机的坐标
    for point in graph.d:
        if point not in graph.points:
            graph.points[point] = np.random.rand(2)

    draw = Drawer(graph, "pre_"+args.output, 'Random Guess:Hypergraph Visualization', True)

    opt = Optimizer(graph)
    graph.points = opt.res
    print(opt.res)

    draw = Drawer(graph, args.output,'Hypergraph Visualization', True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="data.txt", help="name of the data file")
    parser.add_argument("--output", type=str, default="output.png", help="dst dir of output")
    args = parser.parse_args()

    main(args)


