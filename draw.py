# draw.py

import matplotlib.pyplot as plt
import numpy as np
from hypergraph import Hypergraph

class Drawer:
    def __init__(self, graph: Hypergraph, dst: str, name:str):
        
        self.dir = dst

        edges = graph.edges
        vertices = graph.vertices
        points = graph.points
        plt.figure(figsize=(8, 8))

        # 设置不同边的颜色
        colors = plt.cm.viridis(np.linspace(0, 1, len(edges)))

        for i, edge in enumerate(edges):
            # 每条边的顶点集合
            edge_vertices = [points[v] for v in edge]

            # 计算多边形的中心点
            center_x = np.mean([v[0] for v in edge_vertices])
            center_y = np.mean([v[1] for v in edge_vertices])

            # 计算每个顶点到中心点的极坐标角度
            angles = np.arctan2([v[1] - center_y for v in edge_vertices], [v[0] - center_x for v in edge_vertices])

            # 绘制正多边形
            polygon = plt.Polygon(edge_vertices, closed=True, edgecolor=colors[i], facecolor=colors[i], alpha=0.5, linewidth=2)
            plt.gca().add_patch(polygon)

            # 绘制多边形内的顶点并标注序号
            for v in edge:
                index = vertices.index(v)  # 获取顶点在vertices中的索引
                point = points[v]
                plt.plot(point[0], point[1], 'ro', markersize=5)
                plt.text(point[0], point[1], str(index), color='black', ha='center', va='center')

        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(name)
        plt.grid(True)

        # 在这之前添加保存图形的代码
        plt.savefig(self.dir, dpi=300, bbox_inches='tight')

        plt.show()
