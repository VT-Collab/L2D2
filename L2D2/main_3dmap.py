import hydra
from omegaconf import DictConfig
import os, sys

from scripts.recon_3d.teleop import teleop
from scripts.recon_3d.train import train
from scripts.recon_3d.train_corrections import train_corr

@hydra.main(version_base="1.2", config_name='config', config_path='./cfg')
def main(cfg=DictConfig):
    if not os.path.exists('data/{}/'.format(cfg.task)):
        os.makedirs('data/{}/'.format(cfg.task))

    if cfg.get_demo:
        teleop(cfg)

    if cfg.train:
        train(cfg)

    if cfg.corr:
        train_corr(cfg)

if __name__=='__main__':
    main()