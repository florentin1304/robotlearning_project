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
from train import train
from test import test
import shutil
import argparse

#torch.set_default_dtype(torch.float64)


def test_policy_params(x):
    means = x[ : len(x)//2]
    vars = x[len(x)//2 : ]

    print(f"Start training with \nMeans: {means} \nVars: {vars}")

    source_env =  Monitor(gym.make(f'CustomHopper-Gauss-v0'))
    target_env = Monitor(gym.make(f'CustomHopper-target-v0'))

    # run_name = f"{args.algo}_{args.domain}"
    # run_name += f"_{str(args.var).replace('.', '')}"

    ### (!!!) Doesn't touch masses[0] inside 
    masses = source_env.get_parameters()
    num_masses = len(masses)
    assert len(means) == num_masses-1 

    source_env.set_Gaussian_mean_var(means, vars)

    trained_model = train(source_env, "bayrn/model/last_iteration", "ppo")


    result_mean, result_std = test(trained_model, source_env, n_episodes=250)
    print(f"Results on source_env: {result_mean} +- {result_std}")

    result_mean, result_std = test(trained_model, target_env, n_episodes=100)
    print(f"Results on target_env: {result_mean} +- {result_std}")
    
    return result_mean, result_std, trained_model

def initialize_model(train_x, train_y, train_y_var, state_dict=None):
    # define models for objective and constraint
    model = SingleTaskGP(train_x, train_y) #, train_y_var)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
        
    return mll, model

def get_next_points(train_x, train_y, train_y_var, best_y, bounds, n_points):    
    train_x_mean = torch.mean(train_x, dim=0)
    train_x_std = torch.std(train_x, dim=0)
    train_x = (train_x-train_x_mean)/train_x_std


    train_y_mean = torch.mean(train_y, dim=0)
    train_y_std = torch.std(train_y, dim=0)
    train_y = (train_y-train_y_mean)/train_y_std

    
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


def main(args):    
    #                m1,  m2,  m3, std1, std2, std3
    lower_bounds = [0.1, 0.1, 0.1,  0.1,  0.1, 0.1]
    upper_bounds = [10,   10,  10,    2,    2,   2]
    bounds = [lower_bounds, upper_bounds]
    bounds = torch.tensor(bounds)
    book_keeping = []

    if args.load == "":
        # ======= GENERATE INITIAL TRAINING DATA =======   
        print("="*25, "Generating initial points", "="*25)
        train_x = np.zeros((args.n_init_points, len(bounds[0])))
        for i in range(len(bounds[0])):
            train_x[:, i] = np.random.uniform(bounds[0,i], bounds[1,i], args.n_init_points)

        train_y = []
        train_y_var = []
        
        for iteration in range(args.n_init_points):
            print("="*70)
            print("Pre BO point num = ", iteration)
            iteration_checkpoint_filename = f"bayrn/checkpoints/initial_point_{iteration}.ai"

            t_y, t_y_var, model = test_policy_params(train_x[iteration])
            train_y.append(t_y)
            train_y_var.append(t_y_var)
            
            model.save(os.path.join(os.getcwd(), iteration_checkpoint_filename))
            if max(train_y) == train_y[-1]:
                model.save(os.path.join(os.getcwd(),f"bayrn/best_model.ai"))

            book_keeping.append({
                "params": list(train_x[-1]),
                "outcome": t_y,
                "outcome_var": t_y_var,
                "saved_model": iteration_checkpoint_filename
            })
            with open("bayrn/data.json", "w") as datafile:
                json.dump(book_keeping, datafile, indent=2)

        train_x = torch.tensor(train_x)
        train_y = torch.tensor(train_y).reshape(-1,1)
        train_y_var = torch.tensor(train_y_var).reshape(-1,1)
    else:
        # ======= LOAD TRAINING DATA FROM FILE =======   
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

    # ======= BAYESTIAN OPTIMIZATION ITERATIONS =======
    print("="*25, "Start Bayesian Optimization", "="*25)
    for iteration in range(1, args.n_iterations + 1):    
        print("="*70)
        print("Iteration number = ", iteration)
        t0 = time.time()

        
        candidate = get_next_points(train_x, train_y, train_y_var, 0, bounds, 1)
        candidate_x = candidate[0]
        candidate_y, candidate_y_var, model = test_policy_params(candidate_x[0])

        train_x = torch.cat([train_x, candidate_x])
        train_y = torch.cat([train_y, torch.tensor(candidate_y).reshape(-1,1)])
        train_y_var = torch.cat([train_y_var, torch.tensor(candidate_y_var).reshape(-1,1)])

        iteration_checkpoint_filename = f"bayrn/checkpoints/bo_point_{iteration}.ai"
        model.save(os.path.join(os.getcwd(), iteration_checkpoint_filename))
        if max(train_y.tolist()) == train_y[-1]:
            model.save(os.path.join(os.getcwd(),f"bayrn/best_model.ai"))

        book_keeping.append({
            "params": train_x[-1].tolist(),
            "outcome": candidate_y,
            "outcome_var": candidate_y_var,
            "saved_model": iteration_checkpoint_filename
        })
        
        with open("bayrn/data.json", "w") as datafile:
            json.dump(book_keeping, datafile, indent=2)

        i = torch.argmax(train_y)
        print(f"Best found as {i}-th point analysed")
        print(f"With params {train_x[i]}")
        print(f"And outcome {train_y[i]}")

        t = time.time()
        print(f"Got in {t-t0} seconds")    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--n_init_points', default=5, type=int, help='Number of initial points before BO process (if not loading)')
    parser.add_argument('--n_iterations', default=30, type=int, help='Number BO iterations')
    parser.add_argument("--load", type=str, default="", required=False, help="Path to JSON file to load old data")

    parser.add_argument('--verbose', action='store_true', help='Verbose')
    
    args = parser.parse_args()
    
    
    main(args)
    