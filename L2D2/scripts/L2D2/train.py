import numpy as np
import torch
import json
import os, sys
from models import BC_Vis
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt



class LoadData(Dataset):
    def __init__(self, cfg):
        load_dir = 'data/{}/{}/'.format(cfg.task, cfg.alg)
        self.data = []
        num_demos = 1
        for filename in os.listdir(load_dir):
            if not cfg.fine_tune:
                if 'demo' in filename and 'c' not in filename:
                    traj = json.load(open(os.path.join(load_dir, filename), 'r'))
                    for wp in traj:
                        self.data.append(wp)
                    num_demos +=1
            else:
                if 'demo_c' in filename:
                    traj = json.load(open(os.path.join(load_dir, filename), 'r'))
                    for wp in traj:
                        self.data.append(wp)
                    num_demos +=1
        
        print('imported dataset of length ', len(self.data))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index]).to(device='cuda:0')
    


def train_imitation(cfg, save_name):
    torch.manual_seed(np.random.randint(0, 100000))
    train_data = LoadData(cfg)
    in_dim = 16 if cfg.task == 'long_horizon' else 16
    model = BC_Vis(in_dim).to(device='cuda:0')
    LR = 1e-4
    EPOCHS = 1000
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    BATCH_SIZE = int(len(train_data)/10.)
    if cfg.fine_tune:
        model.load_state_dict(torch.load('data/{}/{}/{}.pt'.format(cfg.task, cfg.alg, save_name), weights_only=False))
        optimizer.load_state_dict(torch.load('data/{}/{}/optim_{}.pt'.format(cfg.task, cfg.alg, save_name), weights_only=False))
        save_name += '_ft'
        LR = 1e-4
        EPOCHS = 500
        BATCH_SIZE = 128
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    plt.figure()
    LOSS = []

    for epoch in range(0, EPOCHS + 1):
        for batch, x in enumerate(train_set):
            if len(x) == 1:
                continue
            state = x[:, :9]
            obj_pos = x[:, 9:16]
            action = x[:, 16:23]
            conf = x[:, 23]

            pred, mu, logvar = model(state, obj_pos)
            loss = model.loss(mu, logvar, pred, action, conf)
            LOSS.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch%100 == 0:
            print('Epoch: {}, Loss = {}'.format(epoch,loss.item()))
            torch.save(model.state_dict(), 'data/{}/{}/{}.pt'.format(cfg.task, cfg.alg, save_name, epoch))
            torch.save(optimizer.state_dict(), 'data/{}/{}/optim_{}.pt'.format(cfg.task, cfg.alg, save_name, epoch))

def train_ensemble(cfg):
    for idx in range(cfg.num_ensembles):
        train_imitation(cfg, 'model_{}'.format(idx))
