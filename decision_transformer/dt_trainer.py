import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import gym
import os
from utils import D4RLTrajectoryDataset, set_seed
from dt_model import DecisionTransformer, GPTConfig
import argparse

class Trainer:
    def __init__(self, batch_size, lr, wt_decay, warmup_steps, max_epochs):
        self.batch_size = batch_size
        self.max_epochs = max_epochs

        self.optimizer = torch.optim.AdamW(
                        model.parameters(), 
                        lr=lr, 
                        weight_decay=wt_decay
                    )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda steps: min((steps+1)/warmup_steps, 1)
        )

    def train(self, model, dataset_path, conf, device):
        total_updates = 0

        # training loop
        for epoch in range(self.max_epochs):
            log_action_losses = []
            model.train()

            dataset = D4RLTrajectoryDataset(dataset_path, conf.context_len)
            loader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=True, pin_memory=True,
                                drop_last=True,
                                num_workers=4) 

            for _, (timesteps, states, actions, returns_to_go, traj_mask) in enumerate(loader):

                # reshape data before feeding to model
                timesteps = timesteps.reshape(self.batch_size,conf.context_len).to(device)	# B x T
                states = states.reshape(self.batch_size,conf.context_len,conf.state_dim,conf.state_dim).to(dtype=torch.float32, device=device)			# B x T x state_dim
                actions = actions.reshape(self.batch_size,conf.context_len).to(dtype=torch.float32, device=device)		# B x T x act_dim
                returns_to_go = returns_to_go.reshape(self.batch_size,conf.context_len).to(device) # B x T x 1
                traj_mask = traj_mask.reshape(self.batch_size,conf.context_len).to(device)	# B x T

                # ground truth actions
                action_target = torch.clone(actions).detach().to(device)
                action_target = torch.nn.functional.one_hot(action_target.to(torch.int64), conf.act_dim)

                _, action_preds, _ = model.forward(timesteps=timesteps,
                                                    states=states,
                                                    actions=actions,
                                                    returns_to_go=returns_to_go)

                # only consider non padded elements
                action_preds = action_preds.view(-1, act_dim)[traj_mask.view(-1,) > 0]
                action_target = action_target.type(torch.float32).view(-1, act_dim)[traj_mask.view(-1,) > 0]

                # cross-entropy loss for discrete action, mse for continuous action
                action_loss = F.cross_entropy(action_preds, action_target)

                self.optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                log_action_losses.append(action_loss.detach().cpu().item())
                total_updates += 1

            mean_action_loss = np.mean(log_action_losses)

            log_str = ("=" * 60 + '\n' +
                    "epoch: " + str(epoch) + '\n' +
                    "num of updates: " + str(total_updates) + '\n' +
                    "action loss: " +  format(mean_action_loss, ".5f")
                    )
            print(log_str)

        # save model
        torch.save(model.state_dict(), save_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=int, default=6e-4)
    parser.add_argument('--wt_decay', type=int, default=0.1)
    parser.add_argument('--warmup_steps', type=int, default=512*20)
    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

    # environment parameters
    dataset = "expert"
    env_name = 'Breakout-v0'
    env_d4rl_name = f'breakout-{dataset}-v2'

    # dataset path
    dataset_path = '../data/' + env_d4rl_name + '.pkl'

    # model saving directory
    log_dir = "./dt_runs/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    prefix = "dt_" + env_d4rl_name
    save_model_name =  prefix + "_model" + ".pt"
    save_model_path = os.path.join(log_dir, save_model_name)

    print("=" * 60)
    print("device set to: " + str(device))
    print("dataset path: " + dataset_path)
    print("model save path: " + save_model_path)

    # model and training config
    env = gym.make(env_name)
    state_dim = 84
    act_dim = env.action_space.n # 4

    conf = GPTConfig(state_dim=state_dim, act_dim=act_dim)

    model = DecisionTransformer(state_dim=conf.state_dim,
							act_dim=conf.act_dim,
							n_blocks=conf.n_blocks,
							h_dim=conf.embed_dim,
							context_len=conf.context_len,
							n_heads=conf.n_heads,
							drop_p=conf.dropout_p).to(device)
    
    # start training    
    trainer = Trainer(args.batch_size, args.lr, args.wt_decay, args.warmup_steps, args.epochs)
    trainer.train(model, dataset_path, conf, device)

    print("=" * 60)
    print("finished training!")
    print("saved model at: " + save_model_path)
    print("=" * 60)