"""
Learn rewards from single agent.
"""

# ----------------------------------------------------------------------------------------------------------------------
# imports
import sys
sys.path.append('../.')
import numpy as n
import random
from utils.geometric_tools import *
from env.gridworld import *
from algs.cirl import reward, irl_gda
import visualization.gridworld_vis as gv
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
from einops import rearrange, reduce, repeat, einsum
import copy
import pandas as pd
import argparse
from pathlib import Path
import wandb
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

# ----------------------------------------------------------------------------------------------------------------------
# Parse command line arguments
# default values
true_expert = False
seed = 1
b_value = 0.02
b_str = 'low'
noise = 0.1
nrollouts = 1000
projection = 'l1_ball'
# projection = 'l2_ball'
# projection = 'linf_ball'
# projection = None
# feature_class = 'everywhere'
feature_class = 'boundary_states'
mode = 'constrained'
# mode = 'unconstrained'

# parse arguments
parser = argparse.ArgumentParser(description = 'Setting')
# parser.add_argument('--b')
parser.add_argument('--mode')
parser.add_argument('--noise')
parser.add_argument('--nrollouts')
parser.add_argument('--projection')
parser.add_argument('--seed')
parser.add_argument('--feature_class')
args = parser.parse_args()
# if args.b:
#     if args.b == 'low':
#         b_value = 0.015
#         b_str = args.b
#     elif args.b == 'medium':
#         b_value = 0.03
#         b_str = args.b
#     elif args.b == 'high':
#         b_value = 1e6
#         b_str = args.b
if args.mode:
    mode = str(args.mode)
if args.noise:
    noise = float(args.noise)
if args.nrollouts:
    nrollouts = int(args.nrollouts)
if args.projection:
    projection = args.projection if args.projection is None else str(args.projection)
if args.feature_class:
    feature_class = str(args.feature_class)
if args.seed:
    print(args.seed)
    seed = int(args.seed)

if true_expert:
    load_path = 'data/expert_data/' + str(noise) + '_noise/' + b_str + '_b/'
    save_path = 'data/irl/true_expert/'+str(noise)+'_noise/'+ feature_class \
            + '_features/' +mode+'_mode/'+projection +'/'
    run_name = 'irl_true_expert' + '_nrollouts_' + str(nrollouts)
else:
    load_path = 'data/expert_data/' + str(noise) + '_noise/' + b_str + '_b/'
    save_path = 'data/irl/' + str(nrollouts) + '_rollouts/' + str(noise) + '_noise/' + feature_class \
                + '_features/' + mode + '_mode/' + projection + '/' + str(seed) + '_seed/'
    run_name = 'irl_data' + '_nrollouts_' + str(nrollouts)
Path(load_path).mkdir(parents=True, exist_ok=True)
Path(save_path).mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------------------------------------------------
# Fix random seeds for reproducibility
random.seed(seed)
np.random.seed(seed)

# ----------------------------------------------------------------------------------------------------------------------
# create environment
params = {
    'grid_height': 6,
    'grid_width': 6,
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
if mode == "constrained":
    b = np.ones(k)*b_value
else:
    b = np.ones(k)*1e6
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
beta = 0.01


# ----------------------------------------------------------------------------------------------------------------------
# Hyperparams
run_config_occ = {
    'eta_mu': 1e-2,
    'eta_xi': 5e1,
    'eta_v': 1e1,
    'eta_w': 1e-2,
    'lambda_w': 0.00
}

run_config = {
    'eta_p': (1-env.gamma) /beta * 1.0,
    'eta_xi': 1e1,
    'eta_w': 1e-3,
    'lambda_w': 0.00
}

wandb_logging = True
wandb_sweep = False
logging = True

if wandb_logging:
    wandb.init(project=run_name, config=run_config)

    if wandb_sweep:
        if wandb_sweep:
            run_config['eta_xi'] = float(wandb.config.eta_xi_v)
            run_config['eta_v'] = float(wandb.config.eta_xi_v)  # same step size for xi and v
            run_config['eta_mu'] = float(wandb.config.eta_xi_v) * float(wandb.config.eta_mu_factor)
            run_config['eta_w'] = float(wandb.config.eta_xi_v) * float(wandb.config.eta_mu_factor) * float(wandb.config.eta_w_factor)

print(args)
print('params ', params)
print('bvalue ', b_value)
print('nrollouts ', nrollouts)
print('rewardclass ', projection)
print('mode ', mode)
print('noise', noise)
print('seed', seed)

# ----------------------------------------------------------------------------------------------------------------------
# Initialize reward model
if feature_class == 'boundary_states':
    feature_ids = [0,1,2,3,4,5,6,11,12,17,18,23,24,29,30,31,32,33,34,35]
elif feature_class == 'except_constrained':
    feature_ids = np.arange(0,36)
    ids = np.concatenate([np.nonzero(env.Psi[:,0,0])[0], np.nonzero(env.Psi[:,0,1])[0]])
    feature_ids = np.delete(feature_ids, ids).tolist()
elif feature_class == 'everywhere':
    feature_ids = np.arange(0,36).tolist()
else:
    pass
delete_ids = np.setdiff1d(np.arange(0,36), feature_ids)
n_features = len(feature_ids)
Phi = np.zeros((env.n, env.m, n_features))
for idx, state_id in enumerate(feature_ids):
    Phi[state_id, :, idx] = 1

if true_expert:
    sigma_E = pd.read_csv(load_path + 'sigma_E_soft_true.csv').to_numpy()[:, 0]
    sigma_E = np.delete(sigma_E, delete_ids, axis=0)
else:
    sigma_E = pd.read_csv(load_path + str(nrollouts) + '_rollouts/seed_' + str(seed) + '/sigma_E_soft_data.csv').to_numpy()[:, 0]
    sigma_E = np.delete(sigma_E, delete_ids, axis=0)

# print optimal policy mismatch
policy = pd.read_csv(load_path + 'policy_E_soft.csv').to_numpy()
mu_true = env.policy2stateactionocc(policy)
w_true = np.zeros(n_features)
w_true[feature_ids.index(5)] = 1/2
w_true[feature_ids.index(33)] = 1/2
v_f = env.approx_vector_cost_eval(np.zeros((env.n, Phi.shape[2])), Phi, policy, max_iters = 1e3, tol = 1e-5, logging = False)
w_grad = np.einsum('i,ij->j', env.nu0, v_f) - sigma_E
if projection == 'l1_ball':
    radius = norm(w_true, ord=1)
elif projection == 'l2_ball':
    radius = norm(w_true, ord=2)
elif projection == 'linf_ball':
    radius = max(w_true)
else:
    radius = 1.0
print('optimal feature expectation mismatch', np.max(np.abs(w_grad)), np.max(np.abs(np.einsum('sad,sa->d', Phi, env.policy2stateactionocc(policy)) / (1-env.gamma) - sigma_E)))

# ----------------------------------------------------------------------------------------------------------------------
# Training
policy_irl, xi, values_irl, w = irl_gda(env, beta, run_config['eta_p'], run_config['eta_xi'], run_config['eta_w'], Phi,
            sigma_E, max_iters=1e5, mode='alt_gda', n_v_tot_eval_steps=50,
            n_v_c_eval_steps=50, n_v_f_eval_steps=50, check_steps=10000, wandb_log=wandb_logging, projection=projection,
                                        radius=radius, logging=logging, mu_true=mu_true, w_true=w_true)

# mu2, xi2, v2, w2 = irl_gda_occ(env, beta, run_config['eta_mu'], run_config['eta_xi'], run_config['eta_v'], run_config['eta_w'],
#                            Phi, sigma_E, run_config['lambda_w'], max_iters=1e6, mode='extragradient',
#                 logging=logging, check_steps=1e4, wandb_log=wandb_logging, projection=projection, threshold=float('inf'),
#                            mu_true=mu_true, w_true=w_true)
#
# policy_irl2 = env.occ2policy(mu2)

# -----------------------------------------------------------------------------------------------------------------------
# Store to csv
pd.DataFrame(policy_irl).to_csv(save_path + "policy.csv", index=False)
pd.DataFrame(w).to_csv(save_path + "w.csv", index=False)
pd.DataFrame(xi).to_csv(save_path + "xi.csv", index=False)
pd.DataFrame(env.policy2stateactionocc(policy_irl)).to_csv(save_path + "mu.csv", index=False)

# pd.DataFrame(policy_irl2).to_csv(save_path + "policy2.csv", index=False)
# pd.DataFrame(mu2).to_csv(save_path + "mu2.csv", index=False)
# pd.DataFrame(w2).to_csv(save_path + "w2.csv", index=False)
# pd.DataFrame(xi2).to_csv(save_path + "xi2.csv", index=False)
# pd.DataFrame(v2).to_csv(save_path + "v2.csv", index=False)
# ----------------------------------------------------------------------------------------------------------------------
# Illustration
fig = gv.plot_gridworld(env, policy_irl, env.grid_width, env.grid_height, values=reward(w, Phi)[:,0], constraints=constraint_patches)
fig.tight_layout()
fig.savefig(save_path + 'imitation.png')

# fig = gv.plot_gridworld(env, policy_irl2, env.grid_width, env.grid_height, values=reward(w2)[:,0], constraints=constraint_patches)
# fig.tight_layout()
# fig.savefig(save_path + 'imitation2.png')