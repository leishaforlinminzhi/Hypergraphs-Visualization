# hypergraph.py

class Hypergraph:
    def __init__(self, v_list:list, e_list:list):
        if(len(v_list) > 0):
            self.num_v = max(v_list) + 1
        else:
            self.num_v = 0
        self.vertices = v_list # 真正被使用的点
        self.edges = e_list
        self.points = {} # 保存顶点的坐标

    def dataloader(self, filename: str):
        self.edges = []
        self.vertices = set()

        with open(filename, "r") as f:  
            for line in f:
                edge = list(map(int, line.strip().split()))
                self.edges.append(edge)
                self.vertices.update(edge)

        self.vertices = list(self.vertices)
        self.num_v = max(self.vertices) + 1
