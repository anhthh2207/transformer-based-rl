from datetime import datetime
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

dataset = "expert"
rtg_scale = 1000

env_name = 'Breakout-v0'
rtg_target = 5000
env_d4rl_name = f'breakout-{dataset}-v2'

# load data from this file
dataset_path = '../../data/breakout-expert-v2.pkl'

# saves model and csv in this directory
log_dir = "./dt_runs/"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# --------------------------------
# Training
# --------------------------------

start_time = datetime.now().replace(microsecond=0)

start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

prefix = "dt_" + env_d4rl_name

save_model_name =  prefix + "_model" + ".pt"
save_model_path = os.path.join(log_dir, save_model_name)
save_best_model_path = save_model_path[:-3] + "_best.pt"

log_csv_name = prefix + "_log_" + start_time_str + ".csv"
log_csv_path = os.path.join(log_dir, log_csv_name)


csv_writer = csv.writer(open(log_csv_path, 'a', 1))
csv_header = (["duration", "num_updates", "action_loss", 
           "eval_avg_reward", "eval_avg_ep_len", "eval_d4rl_score"])

csv_writer.writerow(csv_header)


print("=" * 60)
print("start time: " + start_time_str)
print("=" * 60)

print("device set to: " + str(device))
print("dataset path: " + dataset_path)
print("model save path: " + save_model_path)
print("log csv save path: " + log_csv_path)

env = gym.make(env_name)
env.observation_space.shape = (84, 84)  # resized gray-scale image

state_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

conf = GPTConfig(state_dim=state_dim, act_dim=act_dim)
train_conf = GPTTrainConfig()

traj_dataset = D4RLTrajectoryDataset(dataset_path, conf.context_len, rtg_scale)

traj_data_loader = DataLoader(traj_dataset,
						batch_size=train_conf.batch_size,
						shuffle=True,
						pin_memory=True,
						drop_last=True) 

data_iter = iter(traj_data_loader)

## get state stats from dataset
# state_mean, state_std = traj_dataset.get_state_stats()

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

max_d4rl_score = -1.0
total_updates = 0

for i_train_iter in range(train_conf.max_train_iters):

    log_action_losses = []
    model.train()

    for _ in range(train_conf.num_updates_per_iter):
        try:
            timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)
        except StopIteration:
            data_iter = iter(traj_data_loader)
            timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)

        timesteps = timesteps.reshape(train_conf.batch_size,conf.context_len).to(device)	# B x T
        states = states.float().reshape(train_conf.batch_size,conf.context_len,conf.state_dim,conf.state_dim).to(device)			# B x T x state_dim
        actions = actions.float().reshape(train_conf.batch_size,conf.context_len).to(device)		# B x T x act_dim
        returns_to_go = returns_to_go.reshape(train_conf.batch_size,conf.context_len).to(device) # B x T x 1
        traj_mask = traj_mask.reshape(train_conf.batch_size,conf.context_len).to(device)	# B x T

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        scheduler.step()

        log_action_losses.append(action_loss.detach().cpu().item())


    mean_action_loss = np.mean(log_action_losses)
    time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)

    total_updates += train_conf.num_updates_per_iter

    log_str = ("=" * 60 + '\n' +
            "time elapsed: " + time_elapsed  + '\n' +
            "num of updates: " + str(total_updates) + '\n' +
            "action loss: " +  format(mean_action_loss, ".5f")
            )

    print(log_str)

    log_data = [time_elapsed, total_updates, mean_action_loss]

    csv_writer.writerow(log_data)

    # save model
    print("saving current model at: " + save_model_path)
    torch.save(model.state_dict(), save_model_path)


print("=" * 60)
print("finished training!")
print("=" * 60)
end_time = datetime.now().replace(microsecond=0)
time_elapsed = str(end_time - start_time)
end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
print("started training at: " + start_time_str)
print("finished training at: " + end_time_str)
print("total training time: " + time_elapsed)
print("max d4rl score: " + format(max_d4rl_score, ".5f"))
print("saved max d4rl score model at: " + save_best_model_path)
print("saved last updated model at: " + save_model_path)
print("=" * 60)