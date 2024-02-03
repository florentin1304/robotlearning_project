import time
import torch
import numpy as np
import gym
import os
import argparse
import multiprocessing

from train import train, make_env, set_seed
from test import test
from env.custom_hopper import *

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

def main(args):
    lrs = [0.03, 0.003, 0.0003, 0.00003]
    gammas = [0.99, 0.90, 0.75, 0.5]

    for lr in lrs:
        for gamma in gammas:
            set_seed(args.seed)
            run_name = f"{args.algo}_{args.domain}_lr{lr}_gamma{gamma}"
            proc_path = os.path.join(os.getcwd(), "hyperparams_tuning")    
            log_dir = os.path.join(proc_path, "logs")
            os.makedirs(log_dir, exist_ok=True)

            eval_env = make_env(args)
            env = Monitor(make_env(args), os.path.join(log_dir, run_name))

            model_folder = "models"
            model_folder = os.path.join(proc_path, model_folder)
            os.makedirs(model_folder, exist_ok=True)

            model_name = run_name
            model_path = os.path.join(model_folder, model_name)


            trained_model = train(env, \
                        eval_env, \
                        model_path=model_path, \
                        algorithm=args.algo, \
                        total_timesteps=args.total_timesteps,\
                        reward_threshold=args.reward_threshold if args.reward_threshold is not None else float('inf'),
                        verbose=args.verbose)


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=319029, type=int, help='Random seed')

    parser.add_argument("--domain", type=str, choices=['source', 'target'], required=True,
                        help="Domain to use: ['source', 'target']")
    
    parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="The total number of samples to train on")
    parser.add_argument('--algo', default='ppo', type=str, choices=['ppo', 'sac'], help='RL Algo [ppo, sac]')
    parser.add_argument('--reward_threshold', required=False, type=float, help="Enable early stopping")
    parser.add_argument('--verbose', action='store_true', help='Verbose')
    
    args = parser.parse_args()

    main(args)
    