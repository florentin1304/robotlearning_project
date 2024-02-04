import time
import torch
import numpy as np
import botorch
from botorch import fit_gpytorch_model
from botorch.models import HeteroskedasticSingleTaskGP, SingleTaskGP
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.optim import optimize_acqf
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.sampling import IIDNormalSampler, SobolQMCNormalSampler

from botorch.exceptions import InputDataWarning
import warnings
warnings.filterwarnings("ignore", category=InputDataWarning)

import gym
import os
import json
from stable_baselines3.common.monitor import Monitor
from train import train, set_seed
from test import test
import shutil
import argparse

from utils import get_exp_scheduler

#torch.set_default_dtype(torch.float64)


def test_policy_params(x, args):
    deltas = np.array(x)

    print(f"Start training with \nDeltas: {deltas}")

    source_env =  Monitor(gym.make(f'CustomHopper-udr-v0'))
    source_env.set_udr_delta(deltas)

    source_env_eval =  Monitor(gym.make(f'CustomHopper-udr-v0'))
    source_env_eval.set_udr_delta(deltas)

    target_env = Monitor(gym.make(f'CustomHopper-target-v0'))


    #train(env, eval_env, model_path, algorithm="ppo", learning_rate = 0.0003, gamma=0.99,
    #       total_timesteps=1_000_000, eval_freq=25_000, n_eval_episodes=100, reward_threshold=float('inf'), verbose=False):
    
    exponential_learning_rate = get_exp_scheduler(0.003, 0.00003)
    trained_model = train(source_env, source_env_eval, "bayrn_udr/model/last_iteration",\
                            total_timesteps=args.train_total_timesteps,\
                            reward_threshold=args.train_reward_threshold,\
                            learning_rate=exponential_learning_rate,
                            gamma=0.99)


    source_result_mean, source_result_std = test(trained_model, source_env, n_episodes=250)
    print(f"Results on source_env: {source_result_mean} +- {source_result_std}")

    result_mean, result_std = test(trained_model, target_env, n_episodes=250)
    print(f"Results on target_env: {result_mean} +- {result_std}")
    
    return result_mean, result_std, source_result_mean, source_result_std, trained_model

def initialize_model(train_x, train_y, train_y_var, state_dict=None):
    # define models for objective and constraint
    model = SingleTaskGP(train_x, train_y) #, train_y_var)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
        
    return mll, model

def get_next_points(train_x, train_y, train_y_var, best_y, bounds, n_points):    
    # Standardized or not ? I guess not
    # train_x_mean = torch.mean(train_x, dim=0)
    # train_x_std = torch.std(train_x, dim=0)
    # train_x = (train_x-train_x_mean)/train_x_std
    # train_y_mean = torch.mean(train_y, dim=0)
    # train_y_std = torch.std(train_y, dim=0)
    # train_y = (train_y-train_y_mean)/train_y_std

    mll, model = initialize_model(train_x, train_y, train_y_var)

    fit_gpytorch_model(mll)

    sampler = SobolQMCNormalSampler(2048)
    qNEI = qNoisyExpectedImprovement(model, train_x, sampler)

    candidates = optimize_acqf(
        acq_function=qNEI,
        bounds=bounds,
        q=n_points,
        num_restarts=100,
        raw_samples=500
    )

    return candidates

def load_data_from_file(args):
    train_x = []
    train_y = []
    train_y_var = []

    with open(args.load) as myfile:
        book_keeping = json.load(myfile)
    
    for iteration in book_keeping:
        print(iteration)
        train_x.append(iteration["params"])
        train_y.append(iteration["outcome"])
        train_y_var.append(iteration["outcome_var"])

    train_x = torch.tensor(train_x)
    train_y = torch.tensor(train_y).reshape(-1,1)
    train_y_var = torch.tensor(train_y_var).reshape(-1,1)

    return book_keeping, train_x, train_y, train_y_var

def add_too_book_keeping(book_keeping, params, source_outcome, source_outcome_var, target_outcome, target_outcome_var, saved_model):
    book_keeping.append({   
                "params": params,
                "source_outcome": source_outcome,
                "source_outcome_var": source_outcome_var,
                "target_outcome": target_outcome,
                "target_outcome_var": target_outcome_var,
                "saved_model": saved_model
            })

    with open("bayrn_udr/data.json", "w") as datafile:
        json.dump(book_keeping, datafile, indent=2)

    return book_keeping

def generate_initial_data(args, function_to_fit, bounds):
    print("="*25, "Generating initial points", "="*25)

    book_keeping = []   
    train_x = np.zeros((args.n_init_points, len(bounds[0])))
    for i in range(len(bounds[0])):
        train_x[:, i] = np.random.uniform(bounds[0,i], bounds[1,i], args.n_init_points)

    train_y = []
    train_y_var = []
    
    for iteration in range(args.n_init_points):
        print("="*70)
        print("Pre BO point num = ", iteration)
        iteration_checkpoint_filename = f"bayrn_udr/checkpoints/initial_point_{iteration}{('_'+args.run_name) if args.run_name is not None else ''}.ai"
        iteration_x = train_x[iteration]

        t_y, t_y_var, source_y, source_y_var, model = function_to_fit(iteration_x)
        train_y.append(t_y)
        train_y_var.append(t_y_var)
        
        model.save(os.path.join(os.getcwd(), iteration_checkpoint_filename))
        if max(train_y) == train_y[iteration]:
            model.save(os.path.join(os.getcwd(),f"bayrn_udr/best_model.ai"))

        book_keeping = add_too_book_keeping(book_keeping, list(iteration_x), source_y, source_y_var, t_y, t_y_var, iteration_checkpoint_filename)
        

    train_x = torch.tensor(train_x)
    train_y = torch.tensor(train_y).reshape(-1,1)
    train_y_var = torch.tensor(train_y_var).reshape(-1,1)

    return book_keeping, train_x, train_y, train_y_var



def main(args):    
    set_seed(args.seed)
    function_to_fit = lambda x: test_policy_params(x, args)

    #                r1,  r2,  r3
    lower_bounds = [0.1, 0.1, 0.1]
    upper_bounds = [10,   10,  10]
    bounds = [lower_bounds, upper_bounds]
    bounds = torch.tensor(bounds)

    if args.load == "":
        # ======= GENERATE INITIAL TRAINING DATA =======   
       book_keeping, train_x, train_y, train_y_var = generate_initial_data(args, function_to_fit, bounds)

    else:
        # ======= LOAD TRAINING DATA FROM FILE =======   
        book_keeping, train_x, train_y, train_y_var = load_data_from_file(args)

    # ======= BAYESTIAN OPTIMIZATION ITERATIONS =======
    print("="*25, "Start Bayesian Optimization", "="*25)
    for iteration in range(1, args.n_iterations + 1):    
        print("="*70)
        print("Iteration number = ", iteration)
        t0 = time.time()

        
        candidate = get_next_points(train_x, train_y, train_y_var, 0, bounds, 1)
        candidate_x = candidate[0]
        candidate_y, candidate_y_var, source_y, source_y_var, model = function_to_fit(candidate_x[0])

        train_x = torch.cat([train_x, candidate_x])
        train_y = torch.cat([train_y, torch.tensor(candidate_y).reshape(-1,1)])
        train_y_var = torch.cat([train_y_var, torch.tensor(candidate_y_var).reshape(-1,1)])

        iteration_checkpoint_filename = f"bayrn_udr/checkpoints/bo_point_{iteration}{('_'+args.run_name) if args.run_name is not None else ''}.ai"
        model.save(os.path.join(os.getcwd(), iteration_checkpoint_filename))
        if max(train_y.tolist()) == train_y[-1]:
            model.save(os.path.join(os.getcwd(),f"bayrn_udr/best_model.ai"))

        book_keeping = add_too_book_keeping(book_keeping, train_x[-1].tolist(), source_y, source_y_var, candidate_y, candidate_y_var, iteration_checkpoint_filename)

        i = torch.argmax(train_y)
        print(f"Best found as {i}-th point analysed")
        print(f"With params {train_x[i]}")
        print(f"And outcome {train_y[i]}")

        t = time.time()
        print(f"Got in {t-t0} seconds")    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=319029, type=int, help='Random seed')
    parser.add_argument('--n_init_points', default=5, type=int, help='Number of initial points before BO process (if not loading)')
    parser.add_argument('--n_iterations', default=30, type=int, help='Number BO iterations')

    parser.add_argument('--train_reward_threshold', required=False, default=float('inf'), type=float, help='Reward threshold for early stopping')
    parser.add_argument('--train_total_timesteps', required=False, default=1_000_000, type=int, help='Total timesteps per training')

    parser.add_argument("--load", type=str, default="", required=False, help="Path to JSON file to load old data")
    parser.add_argument("--run_name", type=str, default="", required=False, help="Extra name for models (for multiple runs)")
    parser.add_argument('--verbose', action='store_true', help='Verbose')
    
    args = parser.parse_args()
    
    
    main(args)
    