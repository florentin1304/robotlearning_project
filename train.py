"""Sample script for training a control policy on the Hopper environment

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between TRPO, PPO, and SAC.
"""
import gym
import os
from env.custom_hopper import *

import argparse
import multiprocessing

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold

def set_seed(seed):
    if seed > 0:
        np.random.seed(seed)

def train(env, model_path, algorithm="ppo", total_timesteps=1_000_000, eval_freq=25_000, n_eval_episodes=100, reward_threshold=float('inf'), verbose=False):
    
    print("Training: ", model_path)
    
    if algorithm.lower() == 'ppo': 
        model = PPO("MlpPolicy", env, verbose=verbose)
    elif algorithm.lower() == 'sac':
        model = SAC("MlpPolicy", env, verbose=verbose)
    else:
        raise Exception(f"Algorithm {algorithm} unknown")

    # Stop training when the model reaches the reward threshold
    stop_train_callback = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
    eval_callback = EvalCallback(env, 
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
        env.set_delta(args.delta, args.perc)
    if args.domain=='Gauss':
        masses = env.get_parameters()[1:]
        num_masses = len(masses)
        vars = args.var * np.ones((num_masses-1,))
        env.set_Gaussian_mean_var(masses, vars)
    
    return env

def main(args):
    set_seed(args.seed)
    # env = make_env(args)
    
    run_name = f"{args.algo}_{args.domain}"
    if args.domain=='udr':
        run_name += f"_{str(args.delta).replace('.', '')}{ '_perc' if args.perc else ''}"
    if args.domain=='Gauss':
        run_name += f"_{str(args.var).replace('.', '')}"


    log_dir = os.path.join(os.getcwd(), "train_logs")
    os.makedirs(log_dir, exist_ok=True)

    cpu = multiprocessing.cpu_count()
    env = SubprocVecEnv([lambda : Monitor(make_env(args), os.path.join(log_dir, run_name+f"_{i}")) for i in range(cpu)])

    if args.verbose:
        print('State space:', env.observation_space)  # state-space
        print('Action space:', env.action_space)  # action-space
        print('Dynamics parameters:', env.get_parameters())  # masses of each link of the Hopper
    
    model_folder = "models"
    model_folder = os.path.join(os.getcwd(), model_folder)
    os.makedirs(model_folder, exist_ok=True)

    model_name = run_name
    model_path = os.path.join(model_folder, model_name)
    trained_model = train(env, \
                        model_path=model_path, \
                        algorithm=args.algo, \
                        total_timesteps=args.total_timesteps,\
                        reward_threshold=args.reward_threshold if args.reward_threshold is not None else float('inf'),
                        verbose=args.verbose)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    ### DOMAIN RANDOMIZATION PARAMETERS
    parser.add_argument("--domain", type=str, choices=['source', 'target', 'udr', "Gauss"], required=True,
                        help="Domain to use: ['source', 'target', 'udr', 'Gauss']")
    parser.add_argument("--delta", type=float, default=1.0, help="If domain=='udr', delta used for the range of randomization")
    parser.add_argument('--perc', action='store_true', help='Delta used as percentage')
    parser.add_argument("--var", type=float, default=1.0, help="If domain=='Gauss', vars used for the range of randomization")

    parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="The total number of samples to train on")
    parser.add_argument('--algo', default='ppo', type=str, choices=['ppo', 'sac'], help='RL Algo [ppo, sac]')
    parser.add_argument('--lr', default=0.0003, type=float, help='Learning rate')
    parser.add_argument('--gamma', default=0.99, type=float, help='Learning rate')
    parser.add_argument('--reward_threshold', required=False, type=float, help="Enable early stopping")
    
    parser.add_argument('--verbose', action='store_true', help='Verbose')
    
    args = parser.parse_args()
    
    
    main(args)