import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import cvxpy as cp
import gurobipy

df = pd.read_csv("/Users/ishaanchansarkar/Downloads/flag.csv")

def sim(i, j):
    s = 0
    if df.iloc[i, 1] == df.iloc[j, 1]:
        s += 10
    if df.iloc[i, 5] == df.iloc[j, 5]:
        s += 20
    for k in range(7):
        if df.iloc[i, k+10] == df.iloc[j, k+10]:
            s += 4
    s += 10 * (df.iloc[i, 19] * df.iloc[j, 19])
    s += df.iloc[i, 22] + df.iloc[j, 22]
    return s

def dist(i, j):
    indlist = [1, 5, 10, 11, 12, 13, 14, 15, 16, 19, 22]
    dist = []
    for ind in indlist:
        dist.append((df.iloc[i, ind] - df.iloc[j, ind])**2)
    return np.sqrt(sum(dist))

similarity = np.zeros((194, 194))
dists = np.zeros((194, 194))

for i in range(194):
    for j in range(194):
        if i != j:
            similarity[i, j] = sim(i, j)
            dists[i, j] = dist(i, j)
        else:
            dists[i, j] = 10

print(dists)

def KNNgraph(n, s):
    weights = np.zeros((194, 194))
    for a in range(194):
        arr = s[a,:]
        tmp = np.argpartition(arr, -n)[-n:]
        ind = tmp[np.argsort((-arr)[tmp])]
        for b in ind:
            weights[a, b] = s[a, b]
            weights[b, a] = s[a, b]
    return weights

def neighborhoodGraph(eps, s):
    weights = np.zeros((194, 194))
    for a in range(194):
        for b in range(194):
            if dists[a, b] < eps:
                weights[a, b] = s[a, b]
                weights[b, a] = s[a, b]
    return weights



def bMatchingGraph(b):
    print(cp.installed_solvers())
    x = cp.Variable((194, 194), boolean=True, name='x')
    objective = sum(dists[a, b] * x[a, b] for a in range(194) for b in range(194))
    constraints = []
    constraints += [x == x.T]
    for s in range(194):
        constraints += [np.sum(x[s, :]) == b]
        constraints += [x[s, s] == 0]

    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver='ECOS_BB',verbose=False)
    print(x)
    print(x.value)
    return x

w = KNNgraph(8, similarity)

class GraphVisualization:

    def __init__(self):
        # visual is a list which stores all
        # the set of edges that constitutes a
        # graph
        self.visual = []

    # addEdge function inputs the vertices of an
    # edge and appends it to the visual list
    def addEdge(self, a, b):
        temp = [a, b]
        self.visual.append(temp)

    # In visualize function G is an object of
    # class Graph given by networkx G.add_edges_from(visual)
    # creates a graph with a given list
    # nx.draw_networkx(G) - plots the graph
    # plt.show() - displays the graph
    def visualize(self):
        G = nx.Graph()
        G.add_edges_from(self.visual)
        nx.draw_networkx(G)
        plt.show()


G = GraphVisualization()

for i in range(194):
    for j in range(194):
        if w[i, j] != 0:
            G.addEdge(i, j)

G.visualize()
