import hydra
from omegaconf import DictConfig
import os, sys

from scripts.L2D2.train import train_ensemble
from scripts.L2D2.eval import eval_imitation
from scripts.L2D2.get_demos import get_demos, refine_demos
from scripts.teleop import teleop, process_demos
from scripts.recon_3d.train_corrections import train_corr

@hydra.main(version_base="1.2", config_name='config', config_path='./cfg')
def main(cfg=DictConfig):
    if not os.path.exists('data/{}/{}/'.format(cfg.task, cfg.alg)):
        os.makedirs('data/{}/{}/'.format(cfg.task, cfg.alg))
    
    # Run this each time after running teleop for respective algorithms
    if cfg.process:
        if cfg.alg == 'l2d2':
            train_corr(cfg)
            process_demos(cfg)
            cfg.corr=True
            refine_demos(cfg)
        else:
            process_demos(cfg)
        exit()

    if cfg.alg == 'common':
        teleop(cfg)

    if cfg.alg == 'l2d2':
        if cfg.get_demo:
            get_demos(cfg)
        if cfg.train_il:
            train_ensemble(cfg)
        if cfg.eval:
            eval_imitation(cfg)
        if cfg.corr:
            teleop(cfg)

if __name__=='__main__':
    main()