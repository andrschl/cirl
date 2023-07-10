"""
Generating the experts.
"""
# ----------------------------------------------------------------------------------------------------------------------
# imports
import sys
import numpy as np
sys.path.append('../.')
from env.gridworld import Gridworld
from algs.cmdp import cmdp_gda, cmdp_gda_occ, regularization
from algs.cirl import empirical_expert_feature_expectation
import visualization.gridworld_vis as gv
import matplotlib
from einops import rearrange, repeat
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import copy
import random
import pandas as pd
import argparse
from pathlib import Path
from scipy.stats import entropy
float_formatter = "{:.8f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

# ----------------------------------------------------------------------------------------------------------------------
# Parse command line arguments
# default values
b_value = 0.02
b_str = 'low'
noise = 0.1
N = 10000
seed = 20

# parse arguments
parser = argparse.ArgumentParser(description = "Hyperparameters to pass")
parser.add_argument('--b')
parser.add_argument('--noise')
parser.add_argument('--N')
parser.add_argument('--seed')
args = parser.parse_args()
if args.b:
    if args.b == 'low':
        b_value = 0.02
        b_str = args.b
    elif args.b == 'high':
        b_value = 1e6
        b_str = args.b
    else:
        b_str = 'low'
if args.noise:
    noise = float(args.noise)
if args.N:
    N = int(args.N)
if args.seed:
    seed = int(args.seed)

data_path = 'data/expert_data/'+str(noise)+'_noise/'+b_str+'_b/'
Path(data_path).mkdir(parents=True, exist_ok=True)
plotting = True

# ----------------------------------------------------------------------------------------------------------------------
# Fix random seeds for reproducibility
print('args: ', args)
random.seed(seed)
np.random.seed(seed)

# ----------------------------------------------------------------------------------------------------------------------
# create environment
params = {
    'grid_height': 6,
    'grid_width' : 6,
    'noise': float(noise),
    'gamma': 0.9
}
n = params['grid_height'] * params['grid_width']
m = 4
k = 2
Psi = np.zeros((params['grid_height'], params['grid_width'], m, k))
Psi[1:5, 1, :, 0] = 1
Psi[2:4, 3, :, 1] = 1
Psi = rearrange(Psi, 'sx sy a k -> (sy sx) a k')
constraint_patches = np.array([[[1,1],[4,1]], [[2,3],[2,1]]])
b = np.ones(k)*b_value
nu0 = np.ones(n) / (n - len(np.nonzero(Psi[:,0,:])[0]))
nu0[np.nonzero(Psi[:,0,:])[0]] = 0.0
r = np.zeros((n, m))
r[5,:] = 1/2
r[33,:] = 1/2
env = Gridworld(**params, constraints = (Psi, b), nu0 =  nu0, r = r)
env.P[5, :, :] = np.zeros_like(env.P[0, :, :])
env.P[5, :, 5] = 1.0
env.P[33, :, :] = np.zeros_like(env.P[0, :, :])
env.P[33, :, 33] = 1.0
beta = 1.0

print('params', params)
print('bvalue', b_value)
print('N', N)
print('noise', noise)
print('seed', seed)



# ----------------------------------------------------------------------------------------------------------------------
# Get expert
# LP solution (as sanity check)
print('-----------')
print('LP solution')
occ_lp, sol = env.lp_solve()
print(sol.message)
print('objective: ', -sol.fun)
policy_lp = env.occ2policy(occ_lp)
print('constraints: ', env.b - np.einsum('jki,jk->i', env.Psi, occ_lp) / (1-env.gamma))

# Approximate solution to regularized problem
eta_p = (1-env.gamma) /beta * 1
eta_xi = 5e1
print('-----------')
print('GDA solution')
policy_soft, xi, values_soft = cmdp_gda(env, beta , eta_p, eta_xi, max_iters=2e4, tol=-1, mode='alt_gda',
                                    n_v_tot_eval_steps=50, n_v_c_eval_steps=50, logging=True, check_steps=1000)
occ_soft = env.policy2stateactionocc(policy_soft)
print('objective: ', np.sum(occ_soft * env.r / (1-env.gamma)))
print('constraints: ', env.b - np.einsum('jki,jk->i', env.Psi, occ_soft) / (1-env.gamma))
print('primal: ', np.sum(occ_soft * env.r / (1-env.gamma)) - regularization(env, occ_soft, beta))
print('dual value: ', np.sum(occ_soft * env.r / (1-env.gamma)) - regularization(env, occ_soft, beta) + np.sum(xi * (env.b - np.einsum('jki,jk->i', env.Psi, occ_soft) / (1-env.gamma))))
#
# # Get approximate solution via convex concave approach
# eta_mu = 1e-2
# eta_xi = 5e1
# eta_v = 5e1
# print('-----------')
# print('convex-concave GDA solution')
# occ_soft2, xi2, v = cmdp_gda_occ(env, beta , eta_mu, eta_xi, eta_v, max_iters=5e5, mode='alt_gda', logging = True, check_steps = 1e4)
# policy_soft2 = env.occ2policy(occ_soft2)
# occ_soft2_ = env.policy2stateactionocc(policy_soft2)
# primal_value = np.sum(occ_soft2_ * env.r / (1-env.gamma))
# c_constraints = env.b - np.einsum('jki,jk->i', env.Psi, occ_soft2_) / (1-env.gamma)
# E = rearrange(np.vstack([np.eye(env.n) for _ in range(env.m)]), '(a s) s_next -> s_next s a',
#               a=env.m)
# v_constraints = np.array([np.sum(np.abs(np.einsum('ijk,jk->i', E - env.gamma*env.P, occ_soft2) - (1-env.gamma) * env.nu0)),
#                          np.sum(np.abs(np.einsum('ijk,jk->i', E - env.gamma*env.P, occ_soft2_) - (1-env.gamma) * env.nu0))])
# occ_soft_diff = np.sum(np.abs(occ_soft2-occ_soft2_))
# print('primal: ', primal_value - regularization(env, occ_soft2_, beta))
# print('dual value: ', primal_value - regularization(env, occ_soft2_, beta) + np.sum(xi2 * c_constraints))
# print('constraints: ', c_constraints)
# print('v constraints: ', v_constraints[0], v_constraints[1])
# print('occ diff: ', occ_soft_diff)
# print('occ diff algs: ', np.sum(np.abs(occ_soft2_ - occ_soft)))
#
# print('primal')

# ----------------------------------------------------------------------------------------------------------------------
# Get feature expectation
n_features = env.n
T = 100
Phi= repeat(np.eye(env.n), 'x y -> x a y', a = env.m)
for seed in range(10):
    for N in [10, 100, 1000]:
        random.seed(seed)
        np.random.seed(seed)
        sigma_E_soft_data = empirical_expert_feature_expectation(env, policy_soft, Phi, N, T)
        # sigma_E_soft2_data = empirical_expert_feature_expectation(env, policy_soft2, Phi, N, T)
        sigma_E_lp_data = empirical_expert_feature_expectation(env, policy_lp, Phi, N, T)
        save_path = data_path + str(N) + '_rollouts/seed_' + str(seed) + '/'
        Path(save_path).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(sigma_E_soft_data, columns=['sigma_E']).to_csv(save_path + "sigma_E_soft_data.csv", index=False)
        # pd.DataFrame(sigma_E_soft2_data, columns=['sigma_E']).to_csv(save_path + "sigma_E_soft2_data.csv", index=False)
        pd.DataFrame(sigma_E_lp_data, columns=['sigma_E']).to_csv(save_path + "sigma_E_lp_data.csv", index=False)

sigma_E_soft_true = np.einsum('jki,jk->i', Phi, occ_soft) / (1-env.gamma)
# sigma_E_soft2_true = np.einsum('jki,jk->i', Phi, occ_soft2) / (1-env.gamma)
sigma_E_lp_true = np.einsum('jki,jk->i', Phi, occ_lp) / (1-env.gamma)

# Store to csv
pd.DataFrame(sigma_E_soft_true, columns=['sigma_E']).to_csv(data_path + "sigma_E_soft_true.csv", index=False)
# pd.DataFrame(sigma_E_soft2_true, columns=['sigma_E']).to_csv(data_path + "sigma_E_soft2_true.csv", index=False)
pd.DataFrame(sigma_E_lp_true, columns=['sigma_E']).to_csv(data_path + "sigma_E_lp_true.csv", index=False)

pd.DataFrame(policy_soft).to_csv(data_path + "policy_E_soft.csv", index=False)
pd.DataFrame(occ_soft).to_csv(data_path + "mu_E_soft.csv", index=False)
pd.DataFrame(xi).to_csv(data_path + "xi.csv", index=False)

# pd.DataFrame(policy_soft2).to_csv(data_path + "policy_E_soft2.csv", index=False)
# pd.DataFrame(occ_soft2_).to_csv(data_path + "mu_E_soft2.csv", index=False)
# pd.DataFrame(xi2).to_csv(data_path + "xi2.csv", index=False)

# pd.DataFrame(np.array([primal_value])).to_csv(data_path + "primal_value.csv", index=False)
# pd.DataFrame(c_constraints).to_csv(data_path + "c_constraints.csv", index=False)
# pd.DataFrame(v_constraints).to_csv(data_path + "v_constraints.csv", index=False)
# pd.DataFrame(np.array([occ_soft_diff])).to_csv(data_path + "occ_soft_diff.csv", index=False)

# ----------------------------------------------------------------------------------------------------------------------
# Plotting
if plotting:
    # extract avg distance to target
    rollout_length_factor = 7

    # star bottom left
    test_nu0 = np.zeros(env.n)
    test_nu0[0] = 1.0

    # test LP greedy
    print('--------')
    coloring = env.r[:,0]
    # coloring = np.maximum(env.state_occupancy(policy_soft),2*(np.sum(env.Psi, axis=(1,2)))>0.0)

    fig_lp = gv.plot_gridworld(env, policy_lp, env.grid_width, env.grid_height, values=coloring, constraints=constraint_patches)
    fig_lp.tight_layout()
    fig_lp.savefig(data_path + 'policy_lp.png')

    # test learned soft-greedy
    print('--------')
    fig_soft = gv.plot_gridworld(env, policy_soft, env.grid_width, env.grid_height, values=env.r[:,0], constraints=constraint_patches)
    fig_soft.tight_layout()
    fig_soft.savefig(data_path + 'policy_soft.png')

    # # test learned soft-greedy 2
    # print('--------')
    # fig_soft = gv.plot_gridworld(env, env.occ2policy(occ_soft2), env.grid_width, env.grid_height, values=env.r[:, 0],
    #                              constraints=constraint_patches)
    # fig_soft.tight_layout()
    # fig_soft.savefig(data_path + 'policy_soft2.png')