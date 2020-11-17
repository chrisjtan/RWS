import numpy as np
import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help="CORe50 root path")
args = parser.parse_args()

if not os.path.exists("../../../projection_datasets"):
    os.mkdir("../../../projection_datasets")

if not os.path.exists("../../../projection_datasets/CORe50_2"):
    os.mkdir("../../../projection_datasets/CORe50_2")
    

info_list = []  # [frame_num, tracking_sequence, category_label]
feature_list = []

frame_num = 0
sequence_num = 0
for tracking_folder in sorted(os.listdir(args.data_path), key=lambda f: int(''.join(filter(str.isdigit, f)))):
    print(tracking_folder)