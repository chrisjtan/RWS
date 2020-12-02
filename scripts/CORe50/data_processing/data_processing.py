import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help="CORe50 root path")
args = parser.parse_args()

model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
lower_model = torch.nn.Sequential(*list(model.children())[:-1])
lower_model.to('cuda')
lower_model.eval()

if not os.path.exists("../../../projection_datasets"):
    os.mkdir("../../../projection_datasets")

if not os.path.exists("../../../projection_datasets/CORe50"):
    os.mkdir("../../../projection_datasets/CORe50")
    os.mkdir("../../../projection_datasets/CORe50/segments")
else:
    shutil.rmtree("../../../projection_datasets/CORe50")
    os.mkdir("../../../projection_datasets/CORe50")
    os.mkdir("../../../projection_datasets/CORe50/segments")
    

info_list = []  # [frame_num, tracking_sequence, category_label]
feature_list = []

seg_num = 0
sequence_num = 0
for tracking_folder in sorted(os.listdir(args.data_path), key=lambda f: int(''.join(filter(str.isdigit, f)))):
    frame_num = 0
    gt_category = int((int(tracking_folder.split('o')[-1]) - 1) / 5)
    for im_name in sorted(os.listdir(os.path.join(args.data_path, tracking_folder))):
        segment = Image.open(os.path.join(args.data_path, tracking_folder, im_name))
        segment.save("../../../projection_datasets/CORe50/segments/segment_%06d.png" % seg_num)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(segment)
        input_batch = input_tensor.unsqueeze(0)
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
        with torch.no_grad():
            feature = lower_model(input_batch)
        feature = np.array(feature.to('cpu')).flatten()
        feature_list.append(feature)
        info_list.append([frame_num, sequence_num, gt_category])
        frame_num += 1
        seg_num += 1
    sequence_num += 1

np.array(feature_list).dump('../../../projection_datasets/CORe50/features.pickle')
np.array(info_list).dump('../../../projection_datasets/CORe50/infos.pickle')
