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
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

def set_seed(seed):
    if seed > 0:
        np.random.seed(seed)

def train(env, model_path, algorithm="ppo", total_timesteps=500_000, es_num_evals_no_improvement=5, verbose=False):
    
    print("Training: ", model_path)
    
    if algorithm.lower() == 'ppo': 
        model = PPO("MlpPolicy", env, verbose=verbose)
    elif algorithm.lower() == 'sac':
        model = SAC("MlpPolicy", env, verbose=verbose)
    else:
        raise Exception(f"Algorithm {algorithm} unknown")

    # Stop training if there is no improvement after more than 3 evaluations
    if es_num_evals_no_improvement != -1:
        stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=es_num_evals_no_improvement-1, min_evals=8, verbose=int(verbose))
    eval_callback = EvalCallback(env, n_eval_episodes=20, eval_freq=10_000, verbose=int(verbose), \
                                 best_model_save_path=model_path, \
                                     callback_after_eval=stop_train_callback if es_num_evals_no_improvement != -1 else None)
    
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)

    if algorithm.lower() == 'ppo': 
        model = PPO.load(os.path.join(model_path, "best_model"))
    elif algorithm.lower() == 'sac':
        model = SAC.load(os.path.join(model_path, "best_model"))
    else:
        raise Exception(f"Algorithm {algorithm} unknown")

    return model

def main(args):
    set_seed(args.seed)
    env = gym.make(f'CustomHopper-{args.domain}-v0')
    
    run_name = f"{args.algo}_{args.domain}"

    if args.domain=='udr':
        run_name += f"_{str(args.delta).replace('.', '')}{ '_perc' if args.perc else ''}"
        env.set_delta(args.delta, args.perc)
    if args.domain=='Gauss':
        run_name += f"_{str(args.var).replace('.', '')}"
        masses = env.get_parameters()[1:]
        num_masses = len(masses)
        vars = args.var * np.ones((num_masses-1,))
        env.set_Gaussian_mean_var(masses, vars)
    

    log_dir = os.path.join(os.getcwd(), "train_logs")
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, os.path.join(log_dir, run_name))

    if args.verbose:
        print('State space:', env.observation_space)  # state-space
        print('Action space:', env.action_space)  # action-space
        print('Dynamics parameters:', env.get_parameters())  # masses of each link of the Hopper
    
    model_folder = "models"
    model_folder = os.path.join(os.getcwd(), model_folder)
    os.makedirs(model_folder, exist_ok=True)

    model_name = run_name
    model_path = os.path.join(model_folder, model_name)

    trained_model = train(env, model_path=model_path, \
                        algorithm=args.algo, \
                        total_timesteps=args.total_timesteps,\
                        es_num_evals_no_improvement=args.es_num_evals_no_improvement,
                        verbose=args.verbose)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    parser.add_argument("--domain", type=str, choices=['source', 'target', 'udr', "Gauss"], required=True,
                        help="Domain to use: ['source', 'target', 'udr', 'Gauss']")
    parser.add_argument("--delta", type=float, default=1.0, help="If domain=='udr', delta used for the range of randomization")
    parser.add_argument('--perc', action='store_true', help='Delta used as percentage')

    parser.add_argument("--var", type=float, default=1.0, help="If domain=='Gauss', vars used for the range of randomization")

    parser.add_argument("--total_timesteps", type=int, default=500_000, help="The total number of samples to train on")
    parser.add_argument('--algo', default='ppo', type=str, choices=['ppo', 'sac'], help='RL Algo [ppo, sac]')
    parser.add_argument('--lr', default=0.0003, type=float, help='Learning rate')
    parser.add_argument('--es_num_evals_no_improvement', default=5, type=int, help="Enable early stopping")
    # parser.add_argument('--gradient_steps', default=-1, type=int, help='Number of gradient steps when policy is updated in sb3 using SAC. -1 means as many as --args.now')
    parser.add_argument('--verbose', action='store_true', help='Verbose')
    
    args = parser.parse_args()
    
    
    main(args)