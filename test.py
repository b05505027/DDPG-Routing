import networkx as nx



edges = [
    (1, 2, {"weight": 4, "traffic": 0}),
    (1, 3, {"weight": 2, "traffic": 0}),
    (2, 3, {"weight": 1, "traffic": 0}),
    (2, 4, {"weight": 5, "traffic": 0}),
    (3, 4, {"weight": 8, "traffic": 0}),
    (3, 5, {"weight": 10, "traffic": 0}),
    (4, 5, {"weight": 2, "traffic": 0}),
    (4, 6, {"weight": 8, "traffic": 0}),
    (5, 6, {"weight": 5, "traffic": 0}),
]
edge_labels = {
    (1, 2): 4,
    (1, 3): 2,
    (2, 3): 1,
    (2, 4): 5,
    (3, 4): 8,
    (3, 5): 10,
    (4, 5): 2,
    (4, 6): 8,
    (5, 6): 5,
}


G = nx.Graph()
for i in range(1, 7):
    G.add_node(i)
G.add_edges_from(edges)

pos = nx.planar_layout(G)

# This will give us all the shortest paths from node 1 using the weights from the edges.
p1 = nx.shortest_path(G,  weight="weight")

# This will give us the shortest path from node 1 to node 6.
p1to6 = nx.shortest_path(G, source=1, target=6, weight="weight")

# This will give us the length of the shortest path from node 1 to node 6.
length = nx.shortest_path_length(G, source=1, target=6, weight="weight")


es = G.edges()
es[2,3]['traffic'] += 12
print(es[2,3]['traffic'])
print("All shortest paths from 1: ", p1[1][6])
print("Shortest path from 1 to 6: ", p1to6)
print("Length of the shortest path: ", length)

for edge in es.data():
    print('source destination traffic', edge[0], edge[1], edge[2]['traffic'])