import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import gym
import os
from utils import D4RLTrajectoryDataset, GPTConfig, GPTTrainConfig
from dt_model import DecisionTransformer

# --------------------------------
# Configuration
# --------------------------------

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
save_best_model_path = save_model_path[:-3] + "_best.pt"

print("=" * 60)
print("device set to: " + str(device))
print("dataset path: " + dataset_path)
print("model save path: " + save_model_path)

env = gym.make(env_name)
state_dim = 84
act_dim = env.action_space.n # 4

# --------------------------------
# Training
# --------------------------------

conf = GPTConfig(state_dim=state_dim, act_dim=act_dim)
train_conf = GPTTrainConfig()

traj_dataset = D4RLTrajectoryDataset(dataset_path, conf.context_len)

traj_data_loader = DataLoader(traj_dataset,
						batch_size=train_conf.batch_size,
						shuffle=True,
						pin_memory=True,
						drop_last=True,
                        num_workers=4) 

# data_iter = iter(traj_data_loader)

model = DecisionTransformer(state_dim=conf.state_dim,
							act_dim=conf.act_dim,
							n_blocks=conf.n_blocks,
							h_dim=conf.embed_dim,
							context_len=conf.context_len,
							n_heads=conf.n_heads,
							drop_p=conf.dropout_p).to(device)

optimizer = torch.optim.AdamW(
					model.parameters(), 
					lr=train_conf.lr, 
					weight_decay=train_conf.wt_decay
				)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lambda steps: min((steps+1)/train_conf.warmup_steps, 1)
)

total_updates = 0

for epoch in range(train_conf.max_epochs):

    log_action_losses = []
    model.train()

    # for _ in range(train_conf.num_updates_per_iter):
    for _, (timesteps, states, actions, returns_to_go, traj_mask) in enumerate(traj_data_loader):
        # try:
        #     timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)
        # except StopIteration:
        #     data_iter = iter(traj_data_loader)
        #     timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)

		# reshape data before feeding to model
        timesteps = timesteps.reshape(train_conf.batch_size,conf.context_len).to(device)	# B x T
        states = states.reshape(train_conf.batch_size,conf.context_len,conf.state_dim,conf.state_dim).to(dtype=torch.float32, device=device)			# B x T x state_dim
        actions = actions.reshape(train_conf.batch_size,conf.context_len).to(dtype=torch.float32, device=device)		# B x T x act_dim
        returns_to_go = returns_to_go.reshape(train_conf.batch_size,conf.context_len).to(device) # B x T x 1
        traj_mask = traj_mask.reshape(train_conf.batch_size,conf.context_len).to(device)	# B x T

		# ground truth actions
        action_target = torch.clone(actions).detach().to(device)
        action_target = torch.nn.functional.one_hot(action_target.to(torch.int64), conf.act_dim)

        state_preds, action_preds, return_preds = model.forward(
                                                        timesteps=timesteps,
                                                        states=states,
                                                        actions=actions,
                                                        returns_to_go=returns_to_go
                                                    )

        # only consider non padded elements
        action_preds = action_preds.view(-1, act_dim)[traj_mask.view(-1,) > 0]
        action_target = action_target.type(torch.float32).view(-1, act_dim)[traj_mask.view(-1,) > 0]

        action_loss = F.cross_entropy(action_preds, action_target)

        optimizer.zero_grad()
        action_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        log_action_losses.append(action_loss.detach().cpu().item())
        total_updates += 1

    mean_action_loss = np.mean(log_action_losses)

    log_str = ("=" * 60 + '\n' +
            "num of updates: " + str(total_updates) + '\n' +
            "action loss: " +  format(mean_action_loss, ".5f")
            )
    print(log_str)

# save model
torch.save(model.state_dict(), save_model_path)


print("=" * 60)
print("finished training!")
print("saved model at: " + save_model_path)
print("=" * 60)