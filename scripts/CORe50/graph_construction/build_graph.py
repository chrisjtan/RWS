import argparse
import numpy as np
from sklearn.cluster import KMeans
from common_utils.functions import f_score
from common_utils.functions import clustering_n_matching, distance_metric_evaluation

parser = argparse.ArgumentParser()
parser.add_argument('--feature_path', type=str, help='path to first phase features')
parser.add_argument('--info_path', type=str, help='path to segments information')
parser.add_argument('--global_K', type=int, help='K value for global clustering')
parser.add_argument('--local_K', type=int, help='K value for viewpoint clustering')
parser.add_argument('--lambda_', type=float, help='the lambda value in weighting function')

args = parser.parse_args()
features = np.load(args.feature_path, allow_pickle=True)
infos = np.load(args.info_path, allow_pickle=True)

kmeans = KMeans(n_clusters=args.global_K, n_init=10, max_iter=100000, tol=1e-10, random_state=0).\
    fit(np.array(features.tolist()))
C = np.array(kmeans.labels_)  # global clusters

frame_match_matrix = np.zeros((len(features), len(features)))
for i in range(len(features)):
    for j in range(i + 1, len(features)):
        if infos[i, 1] == infos[j, 1]:  # skip for frames in the same sequences
            continue
        if C[i] == C[j]:
            frame_match_matrix[i, j] = 1
            frame_match_matrix[j, i] = 1

seq_count = len(np.unique(infos[:, 1]))  # num of tracking sequences

D_neg = np.zeros((seq_count, seq_count))  # cost for not matching
for i in range(len(D_neg)):
    for j in range(i + 1, len(D_neg)):
        if i == j:
            continue
        cost = 0  # the cost for not matching
        seq_i_indices = np.where(infos[:, 1] == i)[0]  # the frame indices in seq i
        seq_j_indices = np.where(infos[:, 1] == j)[0]
        for p in seq_i_indices:
            for q in seq_j_indices:
                if frame_match_matrix[p, q] == 1:
                    # cost += lam * (1/(np.linalg.norm(p-q)))
                    cost += args.lambda_
        D_neg[i, j] = cost
        D_neg[j, i] = cost

# viewpoint matching
seq_feature_matrix = []
for i in range(len(infos)):
    seq = infos[i, 1]
    while not len(seq_feature_matrix) > seq:
        seq_feature_matrix.append([])
    seq_feature_matrix[seq].append(features[i])

seq_feature_matrix = np.array(seq_feature_matrix)

seq_center_matrix = np.array(seq_feature_matrix)
for row in range(len(seq_feature_matrix)):
    kmeans = KMeans(n_clusters=args.local_K, n_init=10, max_iter=100000, tol=1e-10, random_state=0).\
        fit(np.array(seq_feature_matrix[row]))
    seq_center_matrix[row] = kmeans.cluster_centers_

D = np.zeros((seq_count, seq_count))
for row in range(len(D)):
    for col in range(row+1, len(D[row])):
        D[row, col] = clustering_n_matching(seq_center_matrix[row], seq_center_matrix[col])
        D[col, row] = D[row, col]

seq_match_matrix = np.zeros((seq_count, seq_count))

for row in range(len(seq_match_matrix)):
    for col in range(row+1, len(seq_match_matrix[row])):
        w = np.max([D_neg[row, col] - D[row, col], 0])
        seq_match_matrix[row, col] = w
        seq_match_matrix[col, row] = w

seq_match_matrix.dump('CORe50_adj_matrix_%.2f.pickle' % args.lambda_)
