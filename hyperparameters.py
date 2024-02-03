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
from stable_baselines3.common.utils import get_linear_fn, constant_fn

from utils import get_step_scheduler, get_exp_scheduler

def main(args):
    tournaments = 5

    lrs = ["lin_scheduler", "step_scheduler", "exp_scheduler", 0.003, 0.0003, 0.00003]
    gammas = [0.99, 0.90, 0.75, 0.5]

    for lr in lrs:
        for gamma in gammas:
            for t in range(tournaments):
                set_seed(args.seed + t)
                run_name = f"{args.algo}_{args.domain}_lr{lr}_gamma{gamma}_{t}"
                proc_path = os.path.join(os.getcwd(), "hyperparams_tuning")    
                log_dir = os.path.join(proc_path, "logs")
                os.makedirs(log_dir, exist_ok=True)

                eval_env = Monitor(make_env(args))
                env = Monitor(make_env(args), os.path.join(log_dir, run_name))

                model_folder = "models"
                model_folder = os.path.join(proc_path, model_folder)
                os.makedirs(model_folder, exist_ok=True)

                model_name = run_name
                model_path = os.path.join(model_folder, model_name)

                if lr == "lin_scheduler":
                    my_lr = get_linear_fn(0.003, 0.00003, 0.01)
                elif lr == "step_scheduler":
                    my_lr = get_step_scheduler(0.003, 0.5, 0.2)      
                elif lr == "exp_scheduler":
                    my_lr = get_exp_scheduler(0.003, 0.00003)
                else:
                    my_lr = constant_fn(lr)


                trained_model = train(env, \
                            eval_env, \
                            model_path=model_path, \
                            algorithm=args.algo, \

                            learning_rate=my_lr, \
                            gamma=gamma,\

                            total_timesteps=args.total_timesteps,\
                            reward_threshold=args.reward_threshold if args.reward_threshold is not None else float('inf'),
                            verbose=args.verbose)

                test_log_dir = os.path.join(proc_path, "test_logs")
                os.makedirs(test_log_dir, exist_ok=True)

                mean_reward, std_reward = test(trained_model, env, n_episodes=100)

                print("="*35)
                log_file_name = run_name + ".txt"
                log_file_path = os.path.join(test_log_dir, log_file_name)
                file_contents = ""
                file_contents += f"Model: {run_name}" + "\n"
                file_contents += f"Domain tested on: {env.get_name()}" + "\n"
                file_contents += f"Mean reward: {mean_reward}" + "\n"
                file_contents += f"Std reward: {std_reward}"
                
                with open(log_file_path, "w") as f:
                    f.write(file_contents)
                
                print("="*30)
                print(file_contents)


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
    