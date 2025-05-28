import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
        
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        input_dim = 2
        output_dim = 3

        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        self.apply(weights_init_)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        out = self.network(x)
        return out
    
    def loss(self, pred, target):
        return self.loss_fn(pred, target)
    

class BC_Vis(nn.Module):
    def __init__(self, in_dim):
        super(BC_Vis, self).__init__()

        output_dim = 7
        self.model = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(), 
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.mean = nn.Linear(128, output_dim)
        self.logvar = nn.Linear(128, output_dim)
        self.sigmoid = nn.Sigmoid()
        
        self.apply(weights_init_)

        self.loss_mse = nn.MSELoss()
        self.loss_bce = nn.BCELoss()

    def reparameterization(self, mu, logvar):
        eps = torch.randn_like(logvar)
        z = mu + torch.exp(0.5*logvar) * eps
        return z

    def forward(self, x, z):
        state = torch.cat((x, z), dim=-1)
        h = self.model(state)
        mu, logvar = self.mean(h), self.logvar(h)
        out = self.reparameterization(mu, logvar)
        mu[:, -1] = self.sigmoid(mu[:, -1])
        out[:, -1] = self.sigmoid(out[:, -1])
        return out, mu, logvar

    def loss(self, mu, logvar, pred, true, conf):
        loss_mse = self.loss_mse(true[:, :3], pred[:, :3])
        loss_mse += self.loss_mse(true[:, 3:6], pred[:, 3:6])
        loss_bce = self.loss_bce(pred[:, 6], true[:, 6])

        kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
        return loss_mse + 2*loss_bce + 1e-7*kl_loss
    
