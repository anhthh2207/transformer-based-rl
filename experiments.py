import gym
import argparse

def experiment(variant):
    """ Run an experiment with the given arguments.
    """

    game, dataset = variant['game'], variant['dataset']
    model_type = variant['model_type']

    # Initiate the environment
    if game == 'boxing':
        env = gym.make('Boxing-v4')
    elif game == 'alien':
        env = gym.make('Alien-v4')
    elif game == 'breakout':
        env = gym.make('Breakout-v4')
    else:
        raise NotImplementedError
    
    env.reset()
    
    state_dim = env.observation_space.shape # state dimension
    act_dim = env.action_space.n # action dimension

    for i in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, info = env.step(action)

        env.render()

        if terminated:
            env.reset()

        if (i+1) % 100 == 0:
            print(reward)

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='boxing', help='Available games: boxing, alien, breakout')
    parser.add_argument('--dataset', type=str, default='mixed', help='Dataset types: mixed, medium, expert') 
    parser.add_argument('--mode', type=str, default='normal', help = 'normal for standard setting, delayed for sparse')
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='decision_transformer', help='Model options: decision_transformer, trajectory_transformer, conservative_q_learning') 
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='gelu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learnnig_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    # parser.add_argument('--device', type=str, default='cuda')
    # parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    
    args = parser.parse_args()

    experiment(variant=vars(args))