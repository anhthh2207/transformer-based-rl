import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import gym
import os
import argparse
from tqdm import tqdm

from utils import set_seed, AtariEnv
from official_utils import StackedData, get_trajectory, make_action
from official_model import GPT, GPTConfig

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
            context_len = conf.block_size//3
            dataset = StackedData(dataset_path, context_len)
            loader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=True) 
                                # pin_memory=True, num_workers=4) 
            print("="*60)
            
            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader))
            for it, (timesteps, states, actions, returns_to_go) in pbar:

                # reshape data before feeding to model
                timesteps = timesteps.reshape(self.batch_size,1,1).to(dtype=torch.int64, device=device)	# B x T
                states = states.reshape(self.batch_size,context_len,4,state_dim,state_dim).to(dtype=torch.float32, device=device) # B x T x state_dim
                actions = actions.reshape(self.batch_size,context_len,1).to(dtype=torch.long, device=device)
                returns_to_go = returns_to_go.reshape(self.batch_size,context_len,1).to(device) # B x 1 x 1

                logits, loss = model.forward(states = states,
                                            actions = actions,
                                            targets = actions,
                                            rtgs = returns_to_go,
                                            timesteps = timesteps)
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
            eval_reward = self.evaluate(model, context_len)

            print(f"epoch {epoch+1}: train loss {mean_action_loss:.5f}, average eval reward {eval_reward:.5f}, num of updates {total_updates}")
            
            torch.save(model.state_dict(), f"epoch_{epoch+1}_" + save_model_path)

        print("=" * 60)
        print("finished training!")
        print("saved model at: " + f"epoch_{self.max_epochs+1}_" + save_model_path)
    
    def evaluate(self, model, context_len):
        model.eval()
        env = AtariEnv(game='Breakout', stack=True)
        
        cum_reward = 0
        max_episodes = 10
        for i in range(max_episodes):
            env.reset()
            trajectory = {'observations': [], 'actions': [], 'rewards': [], 'steps': []}
            sum_reward = 0
            step = 0
            while True:
                action = make_action(trajectory, model, context_len, device, random=True)
                observation, reward, terminated, info = env.step(action)
                observation = np.array(observation) / 255.
                trajectory = get_trajectory(trajectory, observation, action, reward, step)
                step += 1
                sum_reward += reward

                if terminated:
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

    print("device set to: " + str(device))
    print("dataset path: " + dataset_path)
    print("model save path: " + save_model_path)

    # model and training config
    state_dim = 84
    act_dim = 4

    conf = GPTConfig(vocab_size=act_dim, n_layer=6, n_head=8, n_embd=128, model_type='reward_conditioned', max_timestep=10000)
    model = GPT(conf).to(device)
    
    # start training    
    trainer = Trainer(args.batch_size, args.lr, args.wt_decay, args.warmup_steps, args.epochs)
    trainer.train(model, dataset_path, conf, device)