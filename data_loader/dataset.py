import os
import numpy as np
import PIL.Image as Image
import cv2
from torch.utils.data import Dataset


class TripletDataset(Dataset):
    def __init__(self, x_dataset, triplets_indices):
        self.x_dataset = x_dataset
        self.triplet_indices = triplets_indices

    def __getitem__(self, index):
        x_index, p_index, n_index = self.triplet_indices[index]
        x_feature = self.x_dataset[x_index, 3]
        p_feature = self.x_dataset[p_index, 3]
        n_feature = self.x_dataset[n_index, 3]
        return x_feature, p_feature, n_feature

    def __len__(self):
        return len(self.triplet_indices)


class TripletDatasetWithConfidence(Dataset):
    def __init__(self, features, triplets_indices):
        self.features = features
        self.triplet_indices = triplets_indices

    def __getitem__(self, index):
        x_index, p_index, n_index, c = self.triplet_indices[index]
        x_feature = self.features[int(x_index)]
        p_feature = self.features[int(p_index)]
        n_feature = self.features[int(n_index)]
        return x_feature, p_feature, n_feature, c

    def __len__(self):
        return len(self.triplet_indices)


class SegmentRGB(Dataset):
    def __init__(self, root, transform, split):
        data_set = []
        img_folder = os.path.join(root)
        if split == 'train':
            for i, img_name in enumerate(sorted(os.listdir(img_folder))):
                if i % 2 == 0:
                    img = cv2.imread(os.path.join(img_folder, img_name))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    data_set.append(img)
        elif split == 'test':
            for i, img_name in enumerate(sorted(os.listdir(img_folder))):
                if i % 10 == 1:
                    img = cv2.imread(os.path.join(img_folder, img_name))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    data_set.append(img)
        else:
            print('need to specify split')
            exit(1)
        self.data = np.array(data_set)
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)
