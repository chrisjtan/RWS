import numpy as np
import torch
import tqdm
import os
import cv2
import torch
import argparse
from sklearn.cluster import KMeans
from models.models import ProjectionNet
from common_utils.functions import cluster_evaluation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_projection', action='store_true', help='if use projected feature')
    parser.add_argument('--feature_path', type=str, help='path of features')
    parser.add_argument('--info_path', type=str, help='path of data information')
    parser.add_argument('--model_path', type=str, help='path of trained model')
    parser.add_argument('--cluster_times', type=int, default='20', help='run n times, then take the average results')

    args = parser.parse_args()

    features = np.load(args.feature_path, allow_pickle=True)
    infos = np.load(args.info_path, allow_pickle=True)

    if args.use_projection:  # enable projection layers
        with torch.no_grad():
            model = ProjectionNet()
            model.load_state_dict(torch.load(args.model_path))
            model.eval()

        new_features = []
        for i in range(len(features)):
            with torch.no_grad():
                new_feature = np.array(model(torch.from_numpy(features[i])))
                new_features.append(new_feature)
        features = np.array(new_features)

    pre_label_list = []

    K = len(np.unique(infos[:, 2]))
    for cluster_num in tqdm.trange(args.cluster_times):  # clustering 20 time and take mean
        kmeans = KMeans(n_clusters=K, n_init=10, max_iter=100000, tol=1e-10, random_state=0).fit(np.array(features.tolist()))
        pre_label_list.append(kmeans.labels_)
    gt_labels = infos[:, 2]
    evaluation = cluster_evaluation(gt_labels, pre_label_list)

    print(evaluation)



