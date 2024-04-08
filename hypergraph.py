# hypergraph.py

class Hypergraph:
    def __init__(self, v_list:list, e_list:list):
        if(len(v_list) > 0):
            self.num_v = max(v_list) + 1
        else:
            self.num_v = 0
        self.edges = e_list # 保存边的列表 其中元素为所有边[边中的点]
        self.points = {} # 保存顶点i对应的坐标
        self.R = {} # 保存关系 R[i]={[..],[..],..} 其中每个元素为关联i个点的边 为e_list的一个划分
        self.d = {} # 顶点的度
        self.v = [] # 保存顶点的序号
        self.eofv = {} # eofv[i]=[...] 顶点i关联的所有边

    def dataloader(self, filename: str):
        self.edges = []
        self.vertices = set()

        with open(filename, "r") as f:  
            i = 0
            for line in f:
                edge = list(map(int, line.strip().split()))
                self.edges.append(edge)
                self.vertices.update(edge)
                
                if(len(edge) not in self.R):
                    self.R[len(edge)] = []
                self.R[len(edge)].append(edge)

                for v in edge:
                    if v not in self.d:
                        self.d[v] = 1
                    else:
                        self.d[v] += 1
                    if(v not in self.eofv):
                        self.eofv[v] = []
                    self.eofv[v].append(i)
                i += 1

        self.vertices = list(self.vertices)
        self.R = dict(sorted(self.R.items()))
        self.d = dict(sorted(self.d.items()))
        self.v = list(self.d.keys())

    def copy(self):
        graph = Hypergraph(self.v,self.edges)
        graph.points = self.points.copy()
        graph.d = self.d.copy()
        graph.v = self.v.copy()
        graph.edges = self.edges.copy()
        graph.eofv = self.eofv.copy()
        return graph
