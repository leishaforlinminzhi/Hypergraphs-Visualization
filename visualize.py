# visualize.py

import argparse
import matplotlib.pyplot as plt
import numpy as np
from hypergraph import Hypergraph

def draw(graph: Hypergraph):
    edges = graph.edges
    vertices = graph.vertices
    points = graph.points
    plt.figure(figsize=(8, 8))

    for edge in edges:
        for v in edge:
            if v not in points:
                # 为每个顶点生成随机的坐标
                points[v] = np.random.rand(2)


    # 设置不同边的颜色
    colors = plt.cm.viridis(np.linspace(0, 1, len(edges)))

    for i, edge in enumerate(edges):

        edge_vertices = [points[v] for v in edge]

        # 计算多边形的中心点
        center_x = np.mean([v[0] for v in edge_vertices])
        center_y = np.mean([v[1] for v in edge_vertices])

        # 计算每个顶点到中心点的极坐标角度
        angles = np.arctan2([v[1] - center_y for v in edge_vertices], [v[0] - center_x for v in edge_vertices])

        # 绘制正多边形
        polygon = plt.Polygon(edge_vertices, closed=True, edgecolor=colors[i], facecolor=colors[i], alpha=0.5, linewidth=2)
        plt.gca().add_patch(polygon)

        # 绘制多边形内的顶点
        for v in edge_vertices:
            plt.plot(v[0], v[1], 'ro', markersize=5)


    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Hypergraph Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()


def main(args):
    graph = Hypergraph([],[])
    graph.dataloader(args.filename)

    draw(graph)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="data.txt", help="name of the data file")
    args = parser.parse_args()

    main(args)


