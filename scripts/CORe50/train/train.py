import numpy as np
import torch
import csv
import os
from torch.utils.data import DataLoader
from data_loader.dataset import TripletDatasetWithConfidence
from models.models import ProjectionNet, TripletNet, xavier_initialize
from models.losses import TripletLossWithConfidence
from common_utils.functions import visualization
import tqdm
import argparse

torch.manual_seed(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_path', type=str, help='path to the first phase features')
    parser.add_argument('--triplet_indices_path', type=str, help='path to the sampled triplets')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--max_epoch', type=int, default=30, help='max training epoch')
    parser.add_argument('--margin', type=float, default='0', help='the margin valuer in triplet loss function')
    args = parser.parse_args()

    features = np.load(args.feature_path, allow_pickle=True)

    triplet_indices = np.load(args.triplet_indices_path, allow_pickle=True)
    triplet_indices = np.array(triplet_indices, dtype=float)
    if torch.cuda.is_available():
        device = torch.device("cuda")

    train_loader = DataLoader(dataset=TripletDatasetWithConfidence(features, triplet_indices), batch_size=128,
                              shuffle=True)
    max_epoch = args.max_epoch
    lr = args.learning_rate
    margin = args.margin
    projection_net = ProjectionNet().to(device)
    model = TripletNet(projection_net).to(device)
    model.apply(xavier_initialize)  # apply xavier initialize
    loss_fn = TripletLossWithConfidence(margin=margin)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-5)

    losses = []
    ite = 0
    epoch_num = 0

    if not os.path.exists('logs'):
        os.mkdir('logs')
    for epoch in tqdm.trange(max_epoch):
        model.train()
        for x_batch, p_batch, n_batch, c_batch in train_loader:
            x_batch = x_batch.to(device)
            p_batch = p_batch.to(device)
            n_batch = n_batch.to(device)
            c_batch = c_batch.to(device)
            x_pred, p_pred, n_pred = model(x_batch, p_batch, n_batch)
            loss = loss_fn(x_pred, p_pred, n_pred, c_batch)

            if ite % 10000 == 0:
                losses.append(loss)
            loss.backward()
            optimizer.step()
            ite += 1
            optimizer.zero_grad()
            if ite % 100 == 0:
                torch.save(losses, 'logs/losses.pickle')
                visualization(losses, 'logs/training_viz.png')
        torch.save(projection_net.state_dict(), 'logs/model_epoch_%03d' % epoch_num)
        epoch_num += 1
