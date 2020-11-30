import numpy as np

triplets = np.load('/home/jt/Projects/RWS/scripts/CORe50/train/triplet_indices_random_walk.pickle', allow_pickle=True)

infos = np.load('/home/jt/Projects/RWS/projection_datasets/CORe50/infos.pickle',
                allow_pickle=True)

print(triplets)
TP = 0
FP = 0

for triplet in triplets:
    a = int(triplet[0])
    p = int(triplet[1])
    n = int(triplet[2])
    if infos[a, 2] == infos[p, 2]:
        TP += 1
    else:
        FP += 1

print('TP is: ', TP, 'FP is:', FP, 'rate: %.2f' %(TP/(TP+FP)))