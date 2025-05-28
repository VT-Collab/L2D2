import numpy as np
import json
import torch
import os, sys
from torch.utils.data import DataLoader, Dataset
from models import Model

class LoadData(Dataset):
    def __init__(self, cfg):
        load_dir = 'data/{}/'.format(cfg.task)
        train_data = []
        target_data = []
        for root, _, files in os.walk(load_dir):
            for filename in files:
                if filename == 'img_traj.json':
                    traj = json.load(open(os.path.join(root, filename), 'r'))
                    for wp in traj:
                        train_data.append(wp[:2])

                if filename == 'traj.json':
                    traj = json.load(open(os.path.join(root, filename), 'r'))
                    for wp in traj:
                        target_data.append(wp[:3])
        
        self.data = np.concatenate((train_data, target_data), axis=1)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index])


def train(cfg):
    for idx in range(cfg.num_ensembles):
        LR = 1e-4
        EPOCH = 5000
        model = Model()
        train_data = LoadData(cfg)
        print("imported dataset of length: ", len(train_data))
        BATCH_SIZE = int(len(train_data)/10)
        train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        for epoch in range (EPOCH + 1):
            for batch, x in enumerate(train_set):
                input = x[:, :2]
                target = x[:, 2:]
                pred = model(input)
                loss = model.loss(pred, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch % 100 == 0:
                print("Epoch = {}, Loss = {}".format(epoch, loss.item()))
                torch.save(model.state_dict(), 'data/{}/model_{}.pt'.format(cfg.task, idx))
                torch.save(optimizer.state_dict(), 'data/{}/optim_model_{}.pt'.format(cfg.task, idx))
