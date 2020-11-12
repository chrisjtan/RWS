import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--feature_path', type=str, help='path to first phase features')
parser.add_argument('--info_path', type=str, help='path to segments information')
parser.add_argument('--global_K', type=int, help='K value for global clustering')

args = parser.parse_args()
features = np.load(args.feature_path, allow_pickle=True)

