import numpy as np
import networkx as nx

def action_transform(action: np.ndarray) -> np.ndarray:
    action = action.reshape(-1)
    action = (action + 1) / 2 * 10 + 1
    action = action.astype(int)
    action = action.tolist()
    action = list(map(lambda x: x+(x==0), action))
    return action


# weights/action generation
actions = np.random.uniform(-1, 1, size=7).reshape(1, -1)
weights = action_transform(actions)
# edge declaration
edges = [
    (1, 3, {"weight": weights[0], "traffic": 0, "index": 1}),
    (1, 2, {"weight": weights[1], "traffic": 0, "index": 2}),
    (2, 4, {"weight": weights[2], "traffic": 0, "index": 3}),
    (3, 4, {"weight": weights[3], "traffic": 0, "index": 4}),
    (3, 5, {"weight": weights[4], "traffic": 0, "index": 5}),
    (4, 5, {"weight": weights[5], "traffic": 0, "index": 6}),
    (2, 3, {"weight": weights[6], "traffic": 0, "index": 7}),
]

# create the graph and calculate its shortest paths
G = nx.Graph()
for i in range(1, 6):
    G.add_node(i)
G.add_edges_from(edges)

# This will give us all the shortest paths from any nodes
shortest_paths = nx.shortest_path(G,  weight="weight")


# traffic generation
random_vector = np.random.exponential(10, size=(5))
traffic_matrix = random_vector * random_vector.reshape(-1, 1)
traffic_matrix = traffic_matrix / np.sum(traffic_matrix) * 100 + 1
print("traffic matrix", traffic_matrix)

for i in range(5):
    for j in range(5):
        path = shortest_paths[i+1][j+1]
        for k in range(len(path)-1):
            G.edges[path[k], path[k+1]]['traffic'] += traffic_matrix[i, j]

# print the traffic of each edge


def quantize_traffic(traffic):
    traffic = traffic / 100
    if traffic <= 0.1:
        return 1
    elif traffic <= 0.2:
        return 2
    elif traffic <= 0.3:
        return 3
    else:
        return 4

new_state = np.ones((5,5), dtype=float)
new_state = new_state * -2
print('new state', new_state)
for edge in G.edges.data():
    new_state[edge[0]-1, edge[1]-1] = quantize_traffic(edge[2]['traffic'])

new_state = new_state
print(new_state)



