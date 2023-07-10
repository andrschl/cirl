""""
Implementation of gradient-descent ascent methods for constrained irl (in occupancy measure and policy space)
"""
import numpy as np
from env.gridworld import *
from einops import einsum
import wandb
import math
from sigpy import l1_proj
from algs.l1_projection import euclidean_proj_l1ball
from algs.cmdp import NPG_update, xi_update, mu_update_occ, xi_update_occ, v_update_occ

# utils

def reward(w, Phi):
    """
    Linear reward model.

    :param w: Reward parameter, shape (d,).
    :param Phi: Reward features, shape (n, m, d).
    :return: return r_w = Phi w.
    """
    return einsum(Phi, w, 's a d, d -> s a')

def dist_to_const_line(r_expert, r):
    r = r.flatten()
    r_expert = r_expert.flatten()
    delta_r = r - r_expert
    n = np.ones_like(r)
    t = np.dot(delta_r, n) / np.dot(n,n)
    return np.sqrt(np.dot(r_expert + t*n - r, r_expert + t*n - r))

def empirical_expert_feature_expectation(env, policy, Phi, N, T, nu0=None):
    """
    Get empirical expert feature expectation.

    :param env: Gridworld environment.
    :param policy: Current policy, shape (n,m).
    :param Phi: Reward features, shape (n, m, d).
    :param N: Number of trajectories.
    :param T: Length of each trajectory.
    :param nu0: Initial state distribution.
    :return: Empirical feature expectation with truncation horizon T.
    """

    rollouts, _ = env.rollout(N, T, policy, nu0=nu0)
    sigma_E = np.zeros(Phi.shape[2])
    for traj_idx in range(N):
        for t in range(T):
            sigma_E += env.gamma**t * Phi[int(rollouts[traj_idx, t, 0]), int(rollouts[traj_idx, t, 1]), :]
    return sigma_E / N

# policy space updates

def w_update(env, w, policy, Phi, sigma_E, eta, init_v_f, max_iters=50, tol=1e-5, projection=None, radius=1.0):
    """
    Calculates the projected gradient descent step for the dual variable w

    :param env: Gridworld environment.
    :param w: Reward parameter, shape (d,).
    :param policy: Current policy, shape (n,m).
    :param Phi: Reward features, shape (n, m, d).
    :param sigma_E: Expert feature expectation, shape (d,).
    :param eta: Learning rate.
    :param init_v_f: Initial feature expectation values, shape (n, d).
    :param max_iters: Maximum number of Bellman iterations for values approximation.
    :param tol: Tolerance for value approximation.
    :param projection: Projection type, something in {None, 'l1_ball' , 'l2_ball', 'linf_ball}
    :param radius: Radius for projection.
    :return: w_next, feature expectation values, grad_w
    """

    v_f = env.approx_vector_cost_eval(init_v_f, Phi, policy, max_iters=max_iters, tol=tol, logging=False)
    grad_w = einsum(env.nu0, v_f, 's, s d -> d') - sigma_E
    w_int = w - eta * grad_w
    if projection == 'l1_ball':
        return l1_proj(radius, w_int), v_f, grad_w
        # return euclidean_proj_l1ball(w_int, s=radius), v_f, grad_w
    if projection == 'l2_ball':
        normalization = math.sqrt(np.sum(w_int * w_int)) / radius
        w_next = w_int / normalization if normalization > 1 else w_int
        return w_next, v_f, grad_w
    if projection == 'linf_ball':
        return np.clip(w_int, -radius, radius), v_f, grad_w
    return w_int, v_f, grad_w

# occupancy measure space updates

def w_update_occ(env, w, mu, Phi, sigma_E, eta, projection=None, radius=1.0):
    """
    Calculates the projected gradient descent step for the dual variable w

    :param env: Gridworld environment.
    :param w: Reward parameter, shape (d,).
    :param mu: Occupancy measure, shape (n,m).
    :param Phi: Reward features, shape (n, m, d).
    :param sigma_E: Expert feature expectation, shape (d,).
    :param eta: Learning rate.
    :param projection: Projection type, something in {None, 'l1_ball' , 'l2_ball', 'linf_ball}
    :param radius: Radius for projection.
    :return: w_next, grad_w
    """

    grad_w = einsum(Phi, mu, 's a d, s a -> d')/(1-env.gamma) - sigma_E
    w_int = w - eta * grad_w
    if projection == 'l1_ball':
        return l1_proj(radius, w_int), grad_w
        # return euclidean_proj_l1ball(w_int, s=radius), v_f, grad_w
    if projection == 'l2_ball':
        normalization = math.sqrt(np.sum(w_int * w_int)) / radius
        w_next = w_int / normalization if normalization > 1 else w_int
        return w_next, grad_w
    if projection == 'linf_ball':
        return np.clip(w_int, -radius, radius), grad_w
    return w_int, grad_w

# logging

def log2console_irl(it, env, Phi, policy, policy_next, xi, grad_xi, w, w_next, grad_w, mu_true=None, w_true=None):
    mu = env.policy2stateactionocc(policy)
    mu_next = env.policy2stateactionocc(policy_next)
    print('step: ', it,
          ', primal value: %0.5f' % np.sum(mu_next * reward(w_next, Phi) / (1 - env.gamma)),
          ', mu_delta: %0.5f' % np.sum(np.abs(mu_next - mu)),
          ', xi: ', xi,
          ', constraint-viol: %0.5f' % np.sum(np.clip(-grad_xi, 0.0, float('inf'))),
          ', w delta %0.5f' % np.sum(np.abs(w_next - w)), '\n',
          '           w_norm %0.5f' % np.sum(np.abs(w_next)),
          ', grad w: %0.5f' % np.max(np.abs(grad_w)),
          ', mu error: %0.5f' % np.sum(np.abs(mu_next - mu_true)) if mu_true is not None else '',
          ', r error: %0.5f' % dist_to_const_line(reward(w_true, Phi), reward(w, Phi)) if w_true is not None else '')

def log2console_irl_occ(it, env, Phi, sigma_E, mu, mu_next, mu_avg, v, v_next, grad_v, xi, grad_xi, w, w_next, w_avg, grad_w, mu_true=None, w_true=None):
    print('step: ', it,
          ', primal value: %0.5f' % np.sum(mu * reward(w_next, Phi) / (1 - env.gamma)),
          ', mu_delta: %0.5f' % np.sum(np.abs(mu_next - mu)),
          ', xi: ', xi, ', xi_avg: ', xi_avg,
          ', constraint-viol: %0.5f' % np.sum(np.clip(-grad_xi, 0.0, float('inf'))),
          ', v_delta: %0.5f' % np.sum(np.abs(v_next - v)),
          ', grad_v: %0.5f' % np.sum(np.abs(grad_v)),
          ', w_delta %0.5f' % np.sum(np.abs(w_next - w)), '\n'
                                                          '           w_norm %0.5f' % np.sum(
            np.abs(w_next)),
          ', grad_w: %0.5f' % np.max(np.abs(grad_w)),
          ', mu error: %0.5f' % np.sum(np.abs(mu_next - mu_true)) if mu_true is not None else '',
          ', w error: %0.5f' % dist_to_const_line(reward(w_true, Phi), reward(w, Phi))  if w_true is not None else '', '\n',
          ', primal avg value: %0.5f' % np.sum(mu_avg * reward(w_avg, Phi) / (1 - env.gamma)),
          ', grad_w avg: %0.5f' % np.sum(
              np.abs(np.einsum('sad,sa->d', Phi, mu_avg) / (1 - env.gamma) - sigma_E)),
          ', w avg error: %0.5f' % dist_to_const_line(reward(w_true, Phi), reward(w_avg, Phi)) if w_true is not None else ''
          )

def log2wandb_irl(it, env, Phi, policy, policy_next, xi, grad_xi, w, w_next, grad_w, mu_true=None, w_true=None):
    mu = env.policy2stateactionocc(policy)
    mu_next = env.policy2stateactionocc(policy_next)
    wandb.log({
        'epoch': it,
        'primal value': np.sum(mu * reward(w_next, Phi) / (1 - env.gamma)),
        'mu delta': np.sum(np.abs(mu_next - mu)),
        'xi': xi,
        'constraint violation': np.sum(np.clip(-grad_xi, 0.0, float('inf'))),
        'w delta': np.sum(np.abs(w_next - w)),
        'w norm': np.sum(np.abs(w_next)),
        'grad_w': np.max(np.abs(grad_w))
    })
    if w_true is not None:
        wandb.log({'r error': dist_to_const_line(reward(w_true, Phi), reward(w, Phi))})
    if mu_true is not None:
        wandb.log({'mu error': np.sum(np.abs(mu - mu_true))})

def log2wandb_irl_occ(it, env, Phi, sigma_E, mu, mu_next, mu_avg, v, v_next, grad_v, xi, grad_xi, w, w_next, w_avg, grad_w, mu_true=None, w_true=None):
    wandb.log({
                'epoch': it,
                'primal value': np.sum(mu * reward(w_next) / (1 - env.gamma)),
                'mu delta': np.sum(np.abs(mu_next - mu)),
                'xi': xi,
                'constraint violation': np.sum(np.clip(-grad_xi, 0.0, float('inf'))),
                'v delta': np.sum(np.abs(v_next - v)),
                'grad_v': np.sum(np.abs(grad_v)),
                'w delta': np.sum(np.abs(w_next - w)),
                'w norm': np.sum(np.abs(w_next)),
                'grad_w': np.max(np.abs(grad_w)),
                'primal avg value': np.sum(mu_avg * reward(w_avg) / (1 - env.gamma)),
                'grad_w avg': np.sum(
                    np.abs(np.einsum('sad,sa->d', Phi, mu_avg) / (1 - env.gamma) - sigma_E))
            })
    if w_true is not None:
        wandb.log({'w error': dist_to_const_line(reward(w_true, Phi), reward(w, Phi)),
                   'w avg error': dist_to_const_line(reward(w_true, Phi), reward(w_avg, Phi))})
    if mu_true is not None:
        wandb.log({'mu error': np.sum(np.abs(mu - mu_true))})

# irl methods

def irl_gda(env, beta, eta_p, eta_xi, eta_w, Phi, sigma_E, max_iters=1e4, mode='sim_gda', n_v_tot_eval_steps=50,
            n_v_c_eval_steps=50, n_v_f_eval_steps=50, logging=True, check_steps=1000, wandb_log=False, projection=None,
            radius=1, mu_true=None, w_true=None):
    """
    Gradient descent-ascent for constrained IRL.

    :param env: Gridworld.
    :param beta: Regularization parameter.
    :param eta_p: Policy learning rate.
    :param eta_xi: Xi learning rate.
    :param eta_w: w learning rate.
    :param Phi: Reward features.
    :param sigma_E: Expert feature expectation.
    :param max_iters: Number of iterations.
    :param mode: 'sim_gda' for simultaneous and 'alt_gda' for alternating primal dual updates.
    :param n_v_tot_eval_steps: # Bellman iterations for total value approximation.
    :param n_v_c_eval_steps: # Bellman iterations for constraint cost value approximation.
    :param n_v_f_eval_steps: # Bellman iterations for feature expectation approximation.
    :param logging: Whether to log to console or not.
    :param check_steps: Logging steps.
    :param wandb_log: Whether to log to wandb or not.
    :param projection: Projection type for w, something in {None, 'l1_ball' , 'l2_ball', 'linf_ball}
    :param radius: Radius for w projection.
    :param mu_true: Ground truth mu expert for logging.
    :param w_true: Ground truth w expert for logging.
    :return: Last iterates policy, xi, v_tot, w
    """

    # initialization
    v_tot = np.zeros(env.n)
    v_c = np.zeros((env.n, env.k))
    v_f = np.zeros((env.n, Phi.shape[2]))
    xi = np.zeros_like(env.b)
    w = np.zeros(Phi.shape[2])
    policy = env.soft_v_greedy(v_tot, beta)

    # start gradient descent ascent
    it = 0
    while it < max_iters:

        # primal update
        r_w_xi = reward(w, Phi) - einsum(env.Psi, xi, 's a k, k -> s a')
        policy_next, v_tot = NPG_update(env, policy, r_w_xi, beta, eta_p, v_tot, max_iters=n_v_tot_eval_steps)

        if mode == 'sim_gda':
            # dual update
            xi_next, v_c, grad_xi = xi_update(env, xi, policy, eta_xi, v_c, max_iters=n_v_c_eval_steps)
            w_next, v_f, grad_w = w_update(env, w, policy, Phi, sigma_E, eta_w, v_f,
                                                  max_iters=n_v_f_eval_steps, projection=projection, radius=radius)
        elif mode == 'alt_gda':
            # dual update
            xi_next, v_c, grad_xi = xi_update(env, xi, policy_next, eta_xi, v_c, max_iters=n_v_c_eval_steps)
            w_next, v_f, grad_w = w_update(env, w, policy_next, Phi, sigma_E, eta_w, v_f,
                                                max_iters=n_v_f_eval_steps, projection=projection, radius=radius)
        else:
            print('Non-existent GDA mode specified')
            break

        # logging
        if it % check_steps == 0:
            if logging:
                log2console_irl(it, env, Phi, policy, policy_next, xi, grad_xi, w, w_next, grad_w, mu_true=mu_true,
                                w_true=w_true)
            if wandb_log:
                log2wandb_irl(it, env, Phi, policy, policy_next, xi, grad_xi, w, w_next, grad_w, mu_true=mu_true,
                              w_true=w_true)

        # update some variables
        policy = policy_next
        xi = xi_next
        w = w_next
        it += 1

    return policy, xi, v_tot, w

def irl_gda_occ(env, beta, eta_mu, eta_xi, eta_v, eta_w, Phi, sigma_E, max_iters=1e4, mode='sim_gda',
                logging=True, check_steps=1e4, wandb_log=False, projection=None, radius=1.0,
                mu_true=None, w_true=None):
    """
    Gradient descent-ascent for constrained IRL in occupancy measure space.

    :param env: Gridworld.
    :param beta: Regularization parameter.
    :param eta_mu: Occupancy measure learning rate.
    :param eta_xi: xi learning rate.
    :param eta_v: v learning rate.
    :param eta_w: w learning rate.
    :param Phi: Reward features.
    :param sigma_E: Expert feature expectation.
    :param max_iters: Number of iterations.
    :param mode: 'sim_gda' for simultaneous, 'extragradient' for extragradient, and 'alt_gda' for alternating primal dual updates.
    :param logging: Whether to log to console or not.
    :param check_steps: Logging steps.
    :param wandb_log: Whether to log to wandb or not.
    :param projection: Projection type for w, something in {None, 'l1_ball' , 'l2_ball', 'linf_ball}
    :param radius: Radius for w projection.
    :param mu_true: Ground truth mu expert for logging.
    :param w_true: Ground truth w expert for logging.
    :return: Last iterates mu, xi, v, w and average iterates mu_avg, xi_avg, v_avg, w_avg
    """

    # initialize
    A = env.A_tensor()

    mu = np.ones((env.n, env.m)) / env.n * env.m
    v = np.zeros(env.n)
    xi = np.zeros_like(env.b)
    w = np.zeros(Phi.shape[2])

    mu_avg = np.zeros_like(mu)
    v_avg = np.zeros_like(v)
    xi_avg = np.zeros_like(xi)
    w_avg = np.zeros_like(w)

    it = 0
    while it < max_iters:

        # primal update
        r_w_xi = reward(w, Phi) - einsum(env.Psi, xi, 's a k, k -> s a')
        mu_int = mu_update_occ(env, A, mu, mu, r_w_xi, v, beta, eta_mu)

        if mode == 'sim_gda':
            # dual update
            xi_next, grad_xi = xi_update_occ(env, xi, mu, eta_xi)
            v_next, grad_v = v_update_occ(env, A, v, mu, eta_v)
            w_next, grad_w = w_update(env, w, mu, Phi, sigma_E, eta_w, projection=projection, radius=radius)
            mu_next = mu_int

        elif mode == 'extragradient':

            # intermediate extrapolation step
            # dual update
            xi_int, _ = xi_update_occ(env, xi, mu, eta_xi)
            v_int, _ = v_update_occ(env, A, v, mu, eta_v)
            w_int, _ = w_update(env, w, mu, Phi, sigma_E, eta_w, projection=projection, radius=radius)

            # update step
            # primal update
            r_w_xi_int = reward(w_int, Phi) - einsum(env.Psi, xi_int, 's a k, k -> s a')
            mu_next = mu_update_occ(env, A, mu, mu_int, r_w_xi_int, v_int, beta, eta_mu)
            
            # dual update
            xi_next, grad_xi = xi_update_occ(env, xi, mu_int, eta_xi)
            v_next, grad_v = v_update_occ(env, A, v, mu_int, eta_v)
            w_next, grad_w = w_update(env, w, mu_int, Phi, sigma_E, eta_w, projection=projection, radius=radius)
            
        elif mode == 'alt-gda':
            # dual update
            xi_next, grad_xi = xi_update_occ(env, xi, mu_int, eta_xi)
            v_next, grad_v = v_update_occ(env, A, v, mu_int, eta_v)
            w_next, grad_w = w_update(env, w, mu_int, Phi, sigma_E, eta_w, projection=projection, radius=radius)
            mu_next = mu_int

        else:
            print('Non-existent GDA mode specified')
            break

        # logging
        if it % check_steps == 0:
            mu_avg = mu_avg / np.sum(mu_avg) # renormalize for numerics
            if logging:
                log2console_irl_occ(it, env, Phi, sigma_E, mu, mu_next, mu_avg, v, v_next, grad_v, xi, grad_xi, w,
                                    w_next, w_avg, grad_w, mu_true=mu_true, w_true=w_true)
            if wandb_log:
                log2wandb_irl_occ(it, env, Phi, sigma_E, mu, mu_next, mu_avg, v, v_next, grad_v, xi, grad_xi, w, w_next,
                                  w_avg, grad_w, mu_true=mu_true, w_true=w_true)

        # update some variables
        mu = mu_next
        xi = xi_next
        v = v_next
        w = w_next
        it += 1

    return mu, xi, v, w, mu_avg, xi_avg, v_avg, w_avg

