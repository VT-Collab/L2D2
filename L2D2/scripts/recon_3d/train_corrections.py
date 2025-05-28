import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader
import json
import os, sys
from models import Model

class LoadData(Dataset):
    def __init__(self, cfg):
        load_dir = 'data/{}/{}/'.format(cfg.task, cfg.alg)
        train_data = []
        target_data = []
        num_corr = 0

        for filename in os.listdir(load_dir):
            if ('c' in filename) and ('demo' not in filename) and ('obj' 
            not in filename) and ('img' not in filename) and ('png' not in filename) and ('pt' not in filename):
                num_corr += 1
            

        for idx in range(num_corr):
            traj = json.load(open(load_dir + 'c{}.json'.format(idx), 'r'))[1:]
            for wp in traj:
                target_data.append(wp[:3])
            
            traj = json.load(open(load_dir + 'c_0_img_{}.json'.format(idx), 'r'))[1:]
            for wp in traj:
                train_data.append(wp[:2])
        
        self.data = np.concatenate((train_data, target_data), axis=1)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index])


def train_corr(cfg):
    for idx in range(cfg.num_ensembles):
        LR = 1e-4
        EPOCH = 1000
        model = Model()
        model.load_state_dict(torch.load('data/play/model_{}.pt'.format(idx)))
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        optimizer.load_state_dict(torch.load('data/play/optim_model_{}.pt'.format(idx)))

        train_data = LoadData(cfg)
        BATCH_SIZE = int(len(train_data)/10.)
        train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

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
                torch.save(model.state_dict(), 'data/{}/{}/recon_model_{}.pt'.format(cfg.task, cfg.alg, idx))
                torch.save(model.state_dict(), 'data/{}/{}/optim_recon_model_{}.pt'.format(cfg.task, cfg.alg, idx))
        

