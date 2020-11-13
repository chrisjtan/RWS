import numpy as np
import argparse
import tqdm
import random
from numpy.linalg import matrix_power
from common_utils.functions import row_norm, row_norm_min_max

parser = argparse.ArgumentParser()
parser.add_argument('--adj_matrix_path', type=str, help='path the the similarity graph')
parser.add_argument('--distance_matrix_path', type=str, help='path to matrix D, used for soft sampling')
parser.add_argument('--info_path', type=str, help='path to segments information')
args = parser.parse_args()

adj_matrix = np.load(args.adj_matrix_path, allow_pickle=True)
distance_matrix = np.load(args.distance_matrix_path, allow_pickle=True)
infos = np.load(args.info_path, allow_pickle=True)

norm_distance_matrix = row_norm_min_max(distance_matrix)
for row in range(len(norm_distance_matrix)):
    for col in range(len(norm_distance_matrix)):
        norm_distance_matrix[row, col] = max(norm_distance_matrix[row, col], norm_distance_matrix[col, row])

for row in range(len(norm_distance_matrix)):
    for col in range(len(norm_distance_matrix[row])):
        if row == col:
            norm_distance_matrix[row, col] = -1


transition_matrix = row_norm(adj_matrix)

H = 3
lam = 0.9
# p_matrix = matrix_power(transition_matrix, H)

p_matrix = np.array(transition_matrix)
for i in range(H-1):
    discount = pow(lam, i)
    p_matrix = np.matmul(p_matrix, transition_matrix * discount)


p_matrix = row_norm(p_matrix)

n_matrix_init = 1 - p_matrix


for i in range(len(n_matrix_init)):  # negative cannot be itself
    n_matrix_init[i, i] = 0

n_matrix = row_norm(n_matrix_init)


def sample_triplet_random_walk(x_index_, sequence_, p_matrix_, n_matrix_, norm_distance_matrix_, infos_, p_size, n_size):
    p_indices = []
    n_indices = []
    p_n_pairs = []
    # sample positive sequences from outer sequences
    p_sequences = []
    # print(p_matrix[sequence_])
    if np.sum(p_matrix_[sequence_]) != 0:
        for j in range(p_size):  # sample 5 positive from outer sequences
            p_seq = np.random.choice(np.arange(len(p_matrix_[sequence_])), p=p_matrix_[sequence_])
            p_sequences.append(p_seq)
        for j in range(p_size):  # sample 5 positive from inner sequence
            p_sequences.append(sequence)
    else:
        for j in range(p_size):
            p_sequences.append(sequence)
    # sample negative sequences from outer sequences
    n_sequences = []
    for j in range(n_size):  # sample negative
        n_seq = np.random.choice(np.arange(len(n_matrix_[sequence_])), p=n_matrix_[sequence_])
        n_sequences.append(n_seq)

    # sample positive indices
    for s in p_sequences:
        indices_in_s = []  # save indices in current sequence for subsample
        c = norm_distance_matrix_[s, sequence]  # confidence for negative
        if c == -1:
            c = 0
        for index in np.where(infos_[:, 1] == s)[0]:
            if index != x_index_:
                indices_in_s.append(index)
        indices_in_s = random.sample(indices_in_s, 1)  # select 1 sample from each sequence
        for sample in indices_in_s:
            p_indices.append([sample, 1 - c])  # 1-c is the confidence to be positive
    # sample negative indices
    for s in n_sequences:
        indices_in_s = []  # save indices in current sequence for subsample
        c = norm_distance_matrix_[s, sequence]  # confidence for negative
        if c == -1:
            print('error')
            break
        for index in np.where(infos_[:, 1] == s)[0]:
            indices_in_s.append(index)
        indices_in_s = random.sample(indices_in_s, 1)  # select 1 sample from each sequence
        for sample in indices_in_s:
            n_indices.append([sample, c])  # c is the confidence to be negative
    for p_index in p_indices:
        for n_index in n_indices:
            c = min(p_index[1], n_index[1])  # use the smaller confidence
            p_n_pairs.append([p_index[0], n_index[0], c])
    return p_n_pairs


triplet_indices = []
for x_index in tqdm.trange(len(infos)):
    if x_index % 1 == 0:
        sequence = infos[x_index, 1]  # seq num
        p_n_pairs = sample_triplet_random_walk(
            x_index,
            sequence,
            p_matrix,
            n_matrix,
            norm_distance_matrix,
            infos,
            p_size=10,
            n_size=5,)
        for p_n_pair in p_n_pairs:
            triplet_indices.append([x_index, p_n_pair[0], p_n_pair[1], p_n_pair[2]])


triplet_indices = np.array(triplet_indices)
print(len(triplet_indices))
np.array(triplet_indices).dump('triplet_indices_random_walk.pickle')
