"""Sample script for training a control policy on the Hopper environment

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between TRPO, PPO, and SAC.
"""
import gym
import os
from env.custom_hopper import *
import torch
import argparse
import multiprocessing

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold

from utils import get_exp_scheduler

def set_seed(seed):
    if seed > 0:
        
        torch.manual_seed(seed)
        set_random_seed(seed)
        np.random.seed(seed)

def train(env, eval_env, model_path, algorithm="ppo", learning_rate = 0.0003, gamma=0.99,
           total_timesteps=1_000_000, eval_freq=25_000, n_eval_episodes=100, reward_threshold=float('inf'), verbose=False):
    
    print("Training: ", model_path)
    
    if algorithm.lower() == 'ppo': 
        model = PPO("MlpPolicy", env, learning_rate=learning_rate, gamma=gamma, verbose=verbose)
    elif algorithm.lower() == 'sac':
        model = SAC("MlpPolicy", env, learning_rate=learning_rate, gamma=gamma, verbose=verbose)
    else:
        raise Exception(f"Algorithm {algorithm} unknown")

    # Stop training when the model reaches the reward threshold
    stop_train_callback = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
    eval_callback = EvalCallback(eval_env, 
                                n_eval_episodes=n_eval_episodes, 
                                eval_freq=eval_freq, 
                                verbose=1, \
                                best_model_save_path=model_path, \
                                callback_after_eval=stop_train_callback)
    
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)

    if algorithm.lower() == 'ppo': 
        model = PPO.load(os.path.join(model_path, "best_model"))
    elif algorithm.lower() == 'sac':
        model = SAC.load(os.path.join(model_path, "best_model"))
    else:
        raise Exception(f"Algorithm {algorithm} unknown")

    return model

def make_env(args):
    env = gym.make(f'CustomHopper-{args.domain}-v0')

    if args.domain=='udr':
        env.set_udr_delta(args.delta, args.perc)
    if args.domain=='Gauss':
        masses = env.get_parameters()[1:]
        num_masses = len(masses)
        vars = args.var * np.ones((num_masses-1,))
        env.set_Gaussian_mean_var(masses, vars)
    
    return env

def main(args):
    set_seed(args.seed)

    log_dir = os.path.join(os.getcwd(), "train_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    eval_env = make_env(args)
    env = make_env(args)
    
    run_name = f"{args.algo}_{env.get_name()}"

    eval_env = Monitor(eval_env)
    env = Monitor(env, os.path.join(log_dir, run_name))

    model_folder = "models"
    model_folder = os.path.join(os.getcwd(), model_folder)
    os.makedirs(model_folder, exist_ok=True)

    model_name = run_name
    model_path = os.path.join(model_folder, model_name)

    exponential_learning_rate = get_exp_scheduler(0.003, 0.00003)
    trained_model = train(env, \
                        eval_env, \
                        model_path=model_path, \
                        algorithm=args.algo, \
                        learning_rate = exponential_learning_rate, \
                        gamma=0.99,\
                        total_timesteps=args.total_timesteps,\
                        reward_threshold=args.reward_threshold if args.reward_threshold is not None else float('inf'),
                        verbose=args.verbose)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=319029, type=int, help='Random seed')

    ### DOMAIN RANDOMIZATION PARAMETERS
    parser.add_argument("--domain", type=str, choices=['source', 'target', 'udr', "Gauss"], required=True,
                        help="Domain to use: ['source', 'target', 'udr', 'Gauss']")
    parser.add_argument("--delta", type=float, default=1.0, help="If domain=='udr', delta used for the range of randomization")
    parser.add_argument('--perc', action='store_true', help='Delta used as percentage')
    parser.add_argument("--var", type=float, default=1.0, help="If domain=='Gauss', vars used for the range of randomization")

    parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="The total number of samples to train on")
    parser.add_argument('--algo', default='ppo', type=str, choices=['ppo', 'sac'], help='RL Algo [ppo, sac]')
    parser.add_argument('--lr', default=0.0003, type=float, help='Learning rate')
    parser.add_argument('--gamma', default=0.99, type=float, help='Gamma')
    parser.add_argument('--reward_threshold', required=False, type=float, help="Enable early stopping")
    
    parser.add_argument('--verbose', action='store_true', help='Verbose')
    
    args = parser.parse_args()
    
    
    main(args)