"""Sample script for training a control policy on the Hopper environment

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between TRPO, PPO, and SAC.
"""
import gym
import os
from env.custom_hopper import *

import argparse

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy


def set_seed(seed):
    if seed > 0:
        np.random.seed(seed)

def test(model, env, n_episodes=100):
    mean_reward, std_reward = evaluate_policy(
                                    model=model,
                                    env=env,
                                    n_eval_episodes=n_episodes,
                                    deterministic=True,
                                    render=False
                                )
    return mean_reward, std_reward

def main(args):
    set_seed(args.seed)
    env = gym.make(f'CustomHopper-{args.domain}-v0')

    test_log_dir = os.path.join(os.getcwd(), "test_logs")
    os.makedirs(test_log_dir, exist_ok=True)

    if args.verbose:
        print('State space:', env.observation_space)  # state-space
        print('Action space:', env.action_space)  # action-space
        print('Dynamics parameters:', env.get_parameters())  # masses of each link of the Hopper
    
    if args.algo.lower() == 'ppo': 
        model = PPO("MlpPolicy", env, verbose=args.verbose)
    elif args.algo.lower() == 'sac':
        model = SAC("MlpPolicy", env, verbose=args.verbose)
    else:
        raise Exception(f"Algorithm {args.algo} unknown")
    
    model_path = os.path.join(os.getcwd(), args.model)
    model.load(model_path)

    mean_reward, std_reward = test(model, env, n_episodes=args.n_episodes)

    print("="*35)
    log_file_name = args.model.split("/")[-1].split(".")[0] + f"_test_on_{args.domain}" + ".txt"
    log_file_path = os.path.join(test_log_dir, log_file_name)
    file_contents = ""
    file_contents += f"Model: {args.model}" + "\n"
    file_contents += f"Domain tested on: {args.domain}" + "\n"
    file_contents += f"Mean reward: {mean_reward}" + "\n"
    file_contents += f"Std reward: {std_reward}"
    
    with open(log_file_path, "w") as f:
        f.write(file_contents)
    
    print("="*30)
    print(file_contents)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument("--domain", type=str, choices=['source', 'target', 'udr'], required=True,
                        help="Domain to use: ['source', 'target', 'udr']")
    parser.add_argument("--n_episodes", type=int, default=100, help="The total number of samples to train on")
    parser.add_argument('--algo', default='ppo', type=str, choices=['ppo', 'sac'], help='RL Algo [ppo, sac]')
    parser.add_argument("--model", type=str, required=True, help="Path to the testing model")
    
    parser.add_argument('--verbose', action='store_true', help='Verbose')
    
    args = parser.parse_args()
    
    main(args)