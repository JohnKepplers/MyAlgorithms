class WeightedGraph():
    def __init__(self):
        self.matrix = [[]]

    def add_vertex(self, v):
        self.matrix[0].append(v)
        self.matrix.append([float('inf')] * len(self.matrix[0]))
        for i in range(1, len(self.matrix) - 1):
            self.matrix[i].append(float('inf'))

    def add_edge(self, v1, v2, w):
        if self.matrix[self.matrix[0].index(v1) + 1][self.matrix[0].index(v2)] == float('inf'):
            self.matrix[self.matrix[0].index(v1) + 1][self.matrix[0].index(v2)] = w
        else:
            print("Error.")

    def floyd_warshall(self):
        floyd_warshall = [[float('inf') for i in range(len(self.matrix[0]))] for i in range(len(self.matrix[0]))]
        for i in range(len(self.matrix[0])):
            for j in range(len(self.matrix[0])):
                if self.matrix[i + 1][j] != float('inf'):
                    floyd_warshall[i][j] = self.matrix[i + 1][j]
        for k in range(len(floyd_warshall)):
            for i in range(len(floyd_warshall)):
                for j in range(len(floyd_warshall)):
                    floyd_warshall[i][j] = min(floyd_warshall[i][j], floyd_warshall[i][k] + floyd_warshall[k][j])
        return floyd_warshall


if __name__ == '__main__':
    graph = WeightedGraph()
    graph.add_vertex(1)
    graph.add_vertex(2)
    graph.add_vertex(3)
    graph.add_vertex(4)
    graph.add_vertex(5)
    graph.add_edge(1, 2, 2)
    graph.add_edge(2, 4, 6)
    graph.add_edge(4, 2, 6)
    graph.add_edge(2, 3, 5)
    graph.add_edge(4, 3, 3)
    graph.add_edge(3, 4, 3)
    graph.add_edge(4, 5, 8)
    graph.add_edge(5, 4, 6)
    graph.add_edge(3, 5, 7)
    print(graph.floyd_warshall())
