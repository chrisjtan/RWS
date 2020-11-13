import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os

info_path = '/home/jt/Projects/RWS/projection_datasets/CORe50/infos.pickle'
infos = np.load(info_path, allow_pickle=True)
m_path = '/home/jt/Projects/RWS/scripts/CORe50/graph_construction/CORe50_adj_matrix_0.20.pickle'

# compute sequence category list
sequence_category_list = [0 for i in range(len(np.unique(infos[:, 1])))]
for seq in np.unique(infos[:, 1]):
    for info in infos:
        if info[1] == seq:
            sequence_category_list[seq] = info[2]
            break


lam = m_path.split('_')[-1][:4]
lam = float(lam)
matching_matrix = np.load(m_path, allow_pickle=True)

G = nx.Graph()
TP = 0
FP = 0
# for i in range(len(matching_matrix)):
#     G.add_node(i)
for node1 in range(len(matching_matrix)):
    for node2 in range(node1 + 1, len(matching_matrix[node1])):
        if matching_matrix[node1, node2] > 0:
            if sequence_category_list[node1] == sequence_category_list[node2]:
                G.add_edge(node1, node2, color='blue', weight=matching_matrix[node1, node2])
                TP += 1
            else:
                G.add_edge(node1, node2, color='red')
                FP += 1

# attrs = {0: {"attr1": 20, "attr2": "nothing"}, 1: {"attr2": 3}}
attrs = {}
for i in range(len(sequence_category_list)):
    attrs[i] = {"C": sequence_category_list[i]}
nx.set_node_attributes(G, attrs)
edges = G.edges()
colors = [G[u][v]['color'] for u, v in edges]
labels = nx.get_node_attributes(G, "C")
pos = nx.kamada_kawai_layout(G)

plt.figure(figsize=(20, 20))
plt.plot()
plt.title('YCB Video lambda:%.2f  TP:%d  FP:%d' % (lam, TP, FP), fontsize=12)
nx.draw(G, labels=labels, edge_color=colors, node_size=300, font_size=20)
plt.savefig('YCB_Video_%.2f___.png' % lam)
plt.clf()
