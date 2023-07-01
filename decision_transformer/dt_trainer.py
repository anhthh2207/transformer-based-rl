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
from dt_model import DecisionTransformer, GPTConfig
from dt_evaluation import get_trajectory, make_action

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
            model.train()

            dataset = D4RLTrajectoryDataset(dataset_path, conf.context_len)
            loader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=True, pin_memory=True,
                                drop_last=True,
                                num_workers=4) 
            print("="*60)
            
            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader))
            for it, (timesteps, states, actions, returns_to_go) in pbar:

                # reshape data before feeding to model
                timesteps = timesteps.reshape(self.batch_size,conf.context_len).to(device)
                states = states.reshape(self.batch_size,conf.context_len,4,conf.state_dim,conf.state_dim).to(dtype=torch.float32, device=device)
                actions = actions.reshape(self.batch_size,conf.context_len).to(dtype=torch.float32, device=device)
                returns_to_go = returns_to_go.reshape(self.batch_size,conf.context_len).to(device)

                _, loss = model.forward(timesteps=timesteps,
                                        states=states,
                                        actions=actions,
                                        targets=actions,
                                        returns_to_go=returns_to_go)

                loss = loss.mean()
                losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                # update progress bar
                pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}")

                total_updates += 1

            mean_action_loss = np.mean(losses)

            # evaluate model
            eval_reward = self.evaluate(model, conf, device)

            print(f"epoch {epoch+1}: train loss {mean_action_loss:.5f}, eval reward {eval_reward:.5f}, num of updates {total_updates}")

        # save model
        torch.save(model.state_dict(), save_model_path)
    
    def evaluate(self, model, conf, device):
        model.eval()
        env = AtariEnv(game='Breakout', stack=True)
        
        cum_reward = 0
        max_episodes = 10
        for i in range(max_episodes):
            # initiate environment
            observation, info = env.reset(seed=args.seed)
            trajectory = {'observations': [], 'actions': [], 'rewards': [], 'steps': []}
            trajectory['observations'].append(observation)
            trajectory['steps'].append(0)
            trajectory['rewards'].append(0)

            # run episode
            sum_reward = 0
            step = 0
            while True:
                action = make_action(trajectory, model, conf.context_len, device)
                observation, reward, terminated, info = env.step(action)
                observation = np.array(observation) / 255.
                trajectory = get_trajectory(trajectory, observation, action, reward, step)
                step += 1
                sum_reward += reward

                if terminated or step >= 10000:
                    break
            cum_reward += sum_reward
        env.close()
        return cum_reward/max_episodes

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
    dataset_path = '../data/' + env_d4rl_name + '-stacked.pkl'

    # model saving directory
    log_dir = "./dt_runs/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    prefix = "dt_" + env_d4rl_name
    save_model_name =  prefix + "_stacked_model" + ".pt"
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