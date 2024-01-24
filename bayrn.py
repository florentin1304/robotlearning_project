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
from stable_baselines3.bench.monitor import Monitor
from train import train
from test import test

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
    
    return result_mean, result_std

def main():    
    #               m1, m2, m3, std1, std2, std3
    lower_bounds = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    upper_bounds = [10, 10, 10, 2, 2, 2]
    bounds = [lower_bounds, upper_bounds]
    bounds = torch.tensor(bounds)

    NUM_ITERATIONS = 100 

    train_x, train_y, train_y_var = generate_initial_data(n=8, bounds=bounds)
    best_observed = torch.max(train_y)

    for iteration in range(1, NUM_ITERATIONS + 1):    
        print("="*30)
        print("Iteration number = ", iteration)
        t0 = time.time()

        
        candidate = get_next_points(train_x, train_y, train_y_var, 0, bounds, 1)
        candidate_x = candidate[0]
        candidate_y, candidate_y_var = test_policy_params(candidate_x[0])

        train_x = torch.cat([train_x, candidate_x])
        train_y = torch.cat([train_y, torch.tensor(candidate_y).reshape(-1,1)])
        train_y_var = torch.cat([train_y_var, torch.tensor(candidate_y_var).reshape(-1,1)])
        best_observed = torch.max(train_y)

        i = torch.argmax(train_y)
        print(f"{train_x[i]=}")
        print(f"{train_y[i]=}")

        t = time.time()
        print(f"Got in {t-t0} seconds")

def generate_initial_data(n, bounds):
    # generate training data   

    train_x = np.zeros((n, len(bounds[0])))
    for i in range(len(bounds[0])):
        train_x[:, i] = np.random.uniform(bounds[0,i], bounds[1,i], n)

    train_y = []
    train_y_var = []

    for i in range(n):
        t_y, t_y_var = test_policy_params(train_x[i])
        train_y.append(t_y)
        train_y_var.append(t_y_var)

    train_x = torch.tensor(train_x)
    train_y = torch.tensor(train_y).reshape(-1,1)
    train_y_var = torch.tensor(train_y_var).reshape(-1,1)


    return train_x, train_y, train_y_var
    
    
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

    sampler = SobolQMCNormalSampler(1024)
    qNEI = qNoisyExpectedImprovement(model, train_x, sampler)

    candidates = optimize_acqf(
        acq_function=qNEI,
        bounds=bounds,
        q=n_points,
        num_restarts=100,
        raw_samples=500
    )

    return candidates


if __name__ == "__main__":
    main()
    