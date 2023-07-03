import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import gym
import os
import argparse
from tqdm import tqdm

from utils import D4RLTrajectoryDataset, set_seed, AtariEnv
from online_dt_model_stacked import GPT, GPTConfig
from online_finetuning import online_finetuning


class Trainer:
    def __init__(self, lr, wt_decay, warmup_steps):
        self.lr = lr
        self.wt_decay = wt_decay
        self.warmup_steps = warmup_steps


    def train(self, data_path, env, episodes, buffer_size, gradient_iterations, model, save_path, device):
        dataset = D4RLTrajectoryDataset(data_path)
        print(dataset.trajectories[0]['observations'].shape)
        print(dataset.trajectories[0]['actions'].shape)
        print(dataset.trajectories[0]['rewards'].shape)
        self.loss_optimizer = torch.optim.AdamW(
                        model.parameters(), 
                        lr=self.lr, 
                        weight_decay=self.wt_decay
                    )
        self.loss_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.loss_optimizer,
            lambda steps: min((steps+1)/self.warmup_steps, 1)
        )

        self.log_temperature_optimizer = torch.optim.AdamW(
                        [model.log_temperature], 
                        lr=self.lr, 
                        weight_decay=self.wt_decay,
                        betas=[0.9, 0.999]
                    )
        self.log_temperature_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.log_temperature_optimizer,
            lambda steps: min((steps+1)/self.warmup_steps, 1)
        )
        optimizers = (self.loss_optimizer, self.log_temperature_optimizer)
        online_finetuning(model, env, optimizers, dataset.trajectories, episodes, buffer_size, gradient_iterations, save_path, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/breakout-expert-v2.pkl')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wt_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--trained_path', type=str, default='decision_transformer/dt_runs/dt_breakout-expert-v2_stacked_model_5.pt')
    parser.add_argument('--buffer_size', type=int, default=50)
    parser.add_argument('--gradient_iterations', type=int, default=10)
    parser.add_argument('--save_path', type=str, default='online_dt_runs')
    parser.add_argument('--env', type=str, default='Breakout')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_seed(args.seed)

    # create directory to save models
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # model and training config
    state_dim = 84
    act_dim = 4

    conf = GPTConfig(vocab_size=act_dim, n_layer=6, n_head=8, n_embd=128, model_type='reward_conditioned', max_timestep=10000)
    model = GPT(conf).to(device)
    # load pretrained model
    model.load_state_dict(torch.load(args.trained_path)) 

    # create trainer
    trainer = Trainer(args.lr, args.wt_decay, args.warmup_steps)

    # train model
    trainer.train(args.data_path, args.env, args.episodes, args.buffer_size, args.gradient_iterations, model, args.save_path, device)

    # save model
    torch.save(model.state_dict(), os.path.join(args.save_path, 'online_dt.pt'))