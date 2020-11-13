import numpy as np
import cv2
import scipy.spatial
import similaritymeasures
from scipy.optimize import linear_sum_assignment
from sklearn import metrics

from coclust.evaluation.external import accuracy

import matplotlib.pyplot as plt


def get_segment(rgb_img, mask):
    """
    :param rgb_img: the rgb image data
    :param mask: the mask data with same size as the rgb image, 0 is background, 255 is the segment.
    :return: segmentation image
    """
    segment = np.ones(rgb_img.shape) * 255
    segment[np.where(np.all(mask == [255, 255, 255], axis=-1))] = \
        rgb_img[np.where(np.all(mask == [255, 255, 255], axis=-1))]
    return segment


def make_image_square(img, size):
    """
    keep the original size scale when making image square so that size matters when extract features.
    :param img:
    :return:
    """
    size = size
    square_img = np.full((size, size, 3), 255, np.uint8)
    ax, ay = (size - img.shape[1]) // 2, (size - img.shape[0]) // 2
    square_img[ay:img.shape[0] + ay, ax:ax + img.shape[1]] = img
    return square_img


def crop_segment(segment):
    """
    crop the image to make it square
    :param segment: the original segment image to crop
    :param size: the size of the resized segment
    :return: the resized image with segment in center
    """
    indices = np.where(np.any(segment != [255, 255, 255], axis=-1))
    segment = segment[np.min(indices[0]):np.max(indices[0]), np.min(indices[1]):np.max(indices[1])]
    return segment


def mean_feature_distance(features1, features2):
    """
    compute the mean feature of two sequences as the distance of these sequences
    :param features1:
    :param features2:
    :return:
    """
    features1 = np.array(features1)
    features2 = np.array(features2)
    mean_1 = np.mean(features1, axis=0)
    mean_2 = np.mean(features2, axis=0)
    mean_dist = np.linalg.norm((mean_1 - mean_2))

    return mean_dist


def top_k_nearest_feature_distance(features1, features2, k):
    """
    compute the nearest k features' distance as the distance of two sequences
    :param features1:
    :param features2:
    :param k: top k nearest feature in the two sequences
    :return:
    """
    features1 = np.array(features1)
    features2 = np.array(features2)
    dist_matrix = scipy.spatial.distance.cdist(features1, features2, 'euclidean')
    dist_matrix = dist_matrix.flatten()
    if len(dist_matrix) >= k:
        smallest_k = dist_matrix[np.argpartition(dist_matrix, k)[:k]]
        ave = np.mean(smallest_k)
    else:
        mean_1 = np.mean(features1, axis=0)
        mean_2 = np.mean(features2, axis=0)
        ave = np.linalg.norm((mean_1 - mean_2))
    return ave


def sub_sample_match_distance(features1, features2, k):
    """
    sub sample k samples from features1/2, find the best match in features 2/1, return the mean feature distance
    """
    sub_sample_1 = []
    for i in range(0, len(features1), int(len(features1)/k)):
        sub_sample_1.append(features1[i])
    sub_sample_1 = np.array(sub_sample_1)
    dist_matrix_1 = scipy.spatial.distance.cdist(sub_sample_1, features2, 'euclidean')
    min_dist_1 = np.min(dist_matrix_1, axis=1)
    sub_sample_2 = []
    for i in range(0, len(features2), int(len(features2)/k)):
        sub_sample_2.append(features2[i])
    sub_sample_2 = np.array(sub_sample_2)
    dist_matrix_2 = scipy.spatial.distance.cdist(sub_sample_2, features1, 'euclidean')
    min_dist_2 = np.min(dist_matrix_2, axis=1)
    match_dist = np.concatenate((min_dist_1, min_dist_2), axis=0)
    return np.mean(match_dist)


def clustering_n_matching(centers_1, centers_2):
    dist_matrix = scipy.spatial.distance.cdist(centers_1, centers_2, 'euclidean')
    row_id, col_id = linear_sum_assignment(dist_matrix)
    return dist_matrix[row_id, col_id].sum()/len(dist_matrix[row_id, col_id])


def clustering_n_curve_matching(centers_1, centers_2):
    dist_matrix = scipy.spatial.distance.cdist(centers_1, centers_2, 'euclidean')
    row_id, col_id = linear_sum_assignment(dist_matrix)
    mean_dist = dist_matrix[row_id, col_id].sum() / len(centers_1)
    curve_1 = centers_1[row_id]
    curve_2 = centers_2[col_id]
    frechet_dist = similaritymeasures.frechet_dist(curve_1, curve_2)
    print(mean_dist, frechet_dist)
    return mean_dist + frechet_dist


def clustering_n_matching_2(centers_1, centers_2):
    """
    clustering and get centers in features1/2, find the best match in features 2/1, return the mean feature distance
    """
    dist_matrix_1 = scipy.spatial.distance.cdist(centers_1, centers_2, 'euclidean')
    min_dist_1 = np.min(dist_matrix_1, axis=1)
    dist_matrix_2 = scipy.spatial.distance.cdist(centers_2, centers_1, 'euclidean')
    min_dist_2 = np.min(dist_matrix_2, axis=1)
    match_dist = np.concatenate((min_dist_1, min_dist_2), axis=0)
    return np.mean(match_dist)


def visualization(losses, path):
    plt.plot(np.arange(len(losses)), losses)
    plt.savefig(path)
    plt.clf()


def cluster_evaluation(gt_labels, pre_label_list):
    """
    evaluate clustering performance with ACC, ARI, NMI scores
    :param gt_labels: the ground truth labels
    :param pre_label_list:  the predicted labels list
    :return: mean ACC, ARI, NMI
    """
    acc_scores = []
    ari_scores = []
    nmi_scores = []
    for i in range(len(pre_label_list)):
        pre_labels = pre_label_list[i]
        acc_scores.append(accuracy(gt_labels, pre_labels)[0])
        ari_scores.append(metrics.adjusted_rand_score(gt_labels, pre_labels))
        nmi_scores.append(metrics.normalized_mutual_info_score(gt_labels, pre_labels))
    return {'mean acc': np.mean(acc_scores), 'mean ari': np.mean(ari_scores), 'mean nmi': np.mean(nmi_scores)}


def distance_metric_evaluation(distance_matrix, sequence_category_list, beta=1):
    """
    evaluate distance metrics with running threshold
    :param distance_matrix: seq.size x seq.size distance matrix
    :param sequence_category_list: category for each sequence
    :return: f_beta score
    """
    match_distance = []  # the distances of the sequence in the same cluster
    dismatch_distance = []  # the distances of the sequence in different cluster
    for row in range(len(distance_matrix)):
        for col in range(len(distance_matrix[row])):
            if row != col:
                if sequence_category_list[row] == sequence_category_list[col]:  # in the same category
                    match_distance.append(distance_matrix[row, col])
                else:
                    dismatch_distance.append(distance_matrix[row, col])
    max_dist = np.max(distance_matrix)
    min_dist = np.min(np.where(distance_matrix < 0, np.inf, distance_matrix))
    f_scores = []
    threshes = np.linspace(min_dist, max_dist, num=1000)
    tps = []

    for thresh in threshes:
        tp = len([i for i in match_distance if i <= thresh])
        tn = len([i for i in dismatch_distance if i > thresh])
        fp = len([i for i in dismatch_distance if i <= thresh])
        fn = len([i for i in match_distance if i > thresh])
        if tp == 0:
            f_core = 0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f_core = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)
        f_scores.append(f_core)
        tps.append(tp)
        # print(thresh, f_core, tp, fp)
    plt.figure(0)
    plt.rcParams.update({'font.size': 12})
    plt.title('Pos Matching Evaluation - YCB Video')
    plt.plot(threshes, np.array(tps)/len(match_distance), label='TP rate', color='brown', linewidth=3)
    plt.plot(threshes, f_scores, label='F_0.5 scores', linewidth=3)
    plt.xlabel('Threshold')
    plt.legend()
    plt.savefig('pos_matching.png', bbox_inches='tight', dpi=500)
    # f_score_study(threshes, f_scores, tps)
    return threshes[np.argmax(f_scores)], max(f_scores)


def f_score_study(threshes, fs, tps):
    """
    What's the range of thresholds to get 80% performance
    """
    threshes = np.array(threshes)
    fs = np.array(fs)
    tps = np.array(tps)
    max_f = np.max(fs)
    f_bar = max_f * 0.8
    thresh_range = []
    for i in range(len(threshes)):
        if fs[i] > f_bar:
            thresh_range.append(threshes[i])
    print('80% thresh range is from ', min(thresh_range), ' to ', max(thresh_range), 'value:', f_bar)


def f_score(tp, fp, fn, beta):
    if tp == 0:
        f_core = 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        print('TP: ', tp, 'FP: ', fp)
        print('precision: ', precision)
        print('recall: ', recall)
        f_core = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)
    return f_core