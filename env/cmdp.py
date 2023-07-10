"""
MDP implementation
"""

import numpy as np
import numpy.random as rn
from scipy.special import softmax, logsumexp
from scipy.stats import entropy
from scipy.optimize import linprog
from einops import rearrange, repeat, einsum
from typing import List, Tuple, Dict, Optional, Any, Union, Callable
from utils.geometric_tools import unit_vector
import math

class CMDP(object):
    """
    Generic finite state action CMDP.

    Attributes:
        n: # states.
        m: # actions.
        k: # constraints.
        gamma: discount rate of MDP.
        r: Numpy array with reward r(s,a) for each state action pair, shape (n, m).
        P: Numpy array transition_dynamics(s'|s,a) with transition probabilities (s,a)->s',shape (n, m, n) = (s, a, s').
        nu0: Numpy array with initial state distribution, shape (n,).
        Psi: Numpy array with constraint cost, shape (n, m, k).
        b: Numpy array with constraint threshold, shape (k,).
    """

    def __init__(self, n: int, m: int, gamma: float, P: Optional[np.ndarray] = None, nu0: Optional[np.ndarray] = None,
                 r: Optional[np.ndarray] = None, constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> None:

        self.n: int = n
        self.m: int = m
        self.gamma: float = gamma
        self.r = np.zeros((self.n, self.m)) if r is None else r
        if constraints is None:
            self.k = 0
            self.Psi = None
            self.b = None
        else:
            self.k = len(constraints[1])
            self.Psi = constraints[0]
            self.b = constraints[1]
        if P is None:
            self.P = unit_vector(rn.rand(self.n, self.m, self.n), axis=2, ord=1)
        else:
            self.P = P
        if nu0 is None:
            self.nu0 = unit_vector(rn.rand(self.n), ord=1)
        else:
            self.nu0 = nu0

    def __str__(self) -> str:
        return "CMDP(n:{}, m:{}, gamma:{}, k:{})".format(self.n, self.m, self.k, self.gamma)

    # sampling from the MDP
    def rollout(self, N: int, T: int, policy: np.ndarray, nu0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate N trajectories of length T,
        following the given policy. Return trajectories and corresponding total discounted rewards.

        :param N: Number of trajectories.
        :param T: Length of an episode.
        :param policy: Numpy array pi(a|s)_(s,a), shape (n, m).
        :param nu0: Different initial state distribution.
        :return: Trajectories, tot_rewards.
        """

        if nu0 is None:
            nu0 = self.nu0

        trajectories = np.zeros((N, T, 3))
        tot_rewards = np.zeros(N)

        # sample from initial state distribution
        s = rn.choice(self.n, size=N, p=nu0)

        # sample trajectories
        for t in range(T):

            # sample from policy
            a = [rn.choice(self.m, p=policy[s[k], :]) for k in range(N)]

            # sample from dynamics
            s_next = [rn.choice(self.n, p=self.P[s[k], a[k], :]) for k in range(N)]

            r = self.r[s, a]
            tot_rewards += self.gamma**t * r
            trajectories[:, t, :] = np.array([s, a, r]).T
            s = s_next

        return trajectories, tot_rewards

    # ------------------------------------------------------------------------------------------------------------------
    # From here on we include some basic value and policy iteration stuff

    # value and q value iteration steps
    def v_it_step(self, v: np.ndarray, r: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate single value iteration step.

        :param v: Array containing values for each state.
        :return: (T v): Bellman optimality operator applied to values, shape (n,).
        """

        r = self.r if r is None else r
        return np.max(r + self.gamma * einsum(self.P, v, 's a s_next, s_next -> s a'), axis=1)

    def q_it_step(self, q: np.ndarray, r: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate single Q-value iteration step.

        :param q: Current value function, shape (n, m).
        :return: (T q): Bellman optimality operator applied to q, shape (n, m).
        """

        r = self.r if r is None else r
        return r + self.gamma * einsum(self.P, np.max(q, axis=1), 's a s_next, s_next -> s a')

    def soft_v_it_step(self, v: np.ndarray, beta: float, r: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate single soft value iteration step.

        :param v: Current value function, shape (n,).
        :param beta: Regularization parameter > 0.
        :return: (T v): Bellman optimality operator applied to values, shape (n,).
        """

        r = self.r if r is None else r
        return beta * logsumexp((r + self.gamma * einsum(self.P, v, 's a s_next, s_next -> s a')) / beta, axis=1)

    def soft_q_it_step(self, q: np.ndarray, beta: float, r: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate single soft Q-value iteration step.

        :param q: Current value function, shape (n, m).
        :param beta: Regularization parameter > 0.
        :return: (T q): Bellman optimality operator applied to q, shape (n, m).
        """

        r = self.r if r is None else r
        return r + beta*self.gamma*einsum(self.P, logsumexp(q / beta, axis=1), 's a s_next, s_next -> s a')

    def soft_v_opt(self, q: np.ndarray, beta: float) -> np.ndarray:
        return beta * logsumexp(q / beta, axis=1)

    def soft_q(self, v: np.ndarray, r: Optional[np.ndarray] = None) -> np.ndarray:

        r = self.r if r is None else r
        return r + self.gamma*einsum(self.P, v, 's a s_next, s_next -> s a')

    # policy evaluation steps
    def v_eval_step(self, v: np.ndarray, policy: np.ndarray, r: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate single value evaluation step.

        :param v: Current value function, shape (n,).
        :param policy: Policy to evaluate, shape (n,m).
        :return: (T^pi v): Bellman expectation operator applied to values, shape (n,).
        """
        r = self.r if r is None else r
        return np.sum(policy * (r + self.gamma * einsum(self.P, v, 's a s_next, s_next -> s a')), axis=1)

    def vector_cost_eval_step(self, v: np.ndarray, c: np.ndarray, policy: np.ndarray)-> np.ndarray:
        """
        Calculate single vector valued evaluation step.

        :param v: Current value function, shape (n, n_dims).
        :param cost: Vector-valued cost, shape (n, m, n_dims).
        :param policy: Policy to evaluate, shape (n,m).
        :return: (T^pi v): Bellman expectation operator applied to constraint_values, shape (n, n_dims).
        """
        return np.sum(np.expand_dims(policy, axis=2) * (c + self.gamma * einsum(self.P, v, 's a s_next, s_next d -> s a d')), axis=1)

    def q_eval_step(self, q: np.ndarray, policy: np.ndarray, r: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate single Q-value evaluation step.

        :param q: Current Q-value function, shape (n, m).
        :param policy: Policy to evaluate, shape (n,m).
        :return: (T^pi q): Bellman expectation operator applied to q, shape (n, m).
        """
        r = self.r if r is None else r
        return r + self.gamma * einsum(self.P, policy * q, 's a s_next, s_next a_next -> s a')

    def soft_v_eval_step(self, v: np.ndarray, policy: np.ndarray, beta: float, r: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate single soft value evaluation step.

        :param v: Current value function, shape (n,).
        :param policy: Policy to evaluate, shape (n,m).
        :param beta: Regularization parameter > 0.
        :return: (T^pi v): Soft Bellman expectation operator applied to v, shape (n,).
        """
        r = self.r if r is None else r
        H = entropy(policy, axis = 1)
        return np.sum(policy * (r + self.gamma * einsum(self.P, v, 's a s_next, s_next -> s a')), axis=1) + beta * H

    def soft_q_eval_step(self, q: np.ndarray, policy: np.ndarray, beta: float, r: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate single soft Q-value evaluation step.

        :param q: Current Q-value function, shape (n, m).
        :param policy: Policy to evaluate, shape (n,m).
        :param beta: Regularization parameter > 0.
        :return: (T^pi q): Soft Bellman expectation operator applied to q, shape (n, m).
        """
        r = self.r if r is None else r
        H = einsum(self.P, entropy(policy, axis=1), 's a s_next, s_next -> s a')
        return r + beta * self.gamma * H + self.gamma * einsum(self.P, policy * q, 's a s_next, s_next a_next -> s a')

    # value and q evaluation via iterative Bellman updates
    def approx_v_eval(self, v: np.ndarray, policy: np.ndarray, max_iters: float = 50,
                          tol: float = 1e-9, log_steps: int = 5, logging: bool = True,
                          r: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Evaluate value function of policy by iterative Bellman updates.

        :param v: Initial values, shape (n,).
        :policy: Policy, shape (n,m).
        :param max_iters: Max number of iterations.
        :param tol: Error tolerance for stopping.
        :param log_steps: Pause time for logging.
        :return: v: Values, shape (n,).
        """

        it = 0
        error = float("inf")
        while it < max_iters and error > tol:
            it += 1
            new_v = self.v_eval_step(v, policy, r=r)
            error = np.max(abs(new_v - v))
            v = new_v
            if logging and it % log_steps == 0:
                print('step: ', it, ', error: ', error)
        return v

    def approx_q_eval(self, q: np.ndarray, policy: np.ndarray, max_iters: float = 50,
                      tol: float = 1e-9, log_steps: int = 5, logging: bool = True,
                      r: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate optimal Q-value function via Q iteration.

        :param q: Initial Q values, shape (n,m).
        :param max_iters: Max number of iterations.
        :param tol: Error tolerance for stopping.
        :param log_steps: Pause time for logging.
        :return: q: Optimal Q values, shape (n,m).
        """

        it = 0
        error = float("inf")
        while it < max_iters and error > tol:
            it += 1
            new_q = self.q_eval_step(q, policy, r=r)
            error = np.max(abs(new_q - q))
            q = new_q
            if logging and it % log_steps == 0:
                print('step: ', it, ', error: ', error)
        return q

    def approx_soft_v_eval(self, v: np.ndarray, policy: np.ndarray, beta: float, max_iters: float = 50,
                               tol: float = 1e-9, log_steps: int = 5, logging: bool = True,
                               r: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate optimal soft value function via soft value iteration.

        :param v: Initial values.
        :param beta: Regularization parameter > 0.
        :param max_iters: Max number of iterations.
        :param tol: Error tolerance for stopping.
        :param log_steps: Pause time for logging.
        :return: v: Optimal values.
        """

        it = 0
        error = float("inf")
        while it < max_iters and error > tol:
            it += 1
            new_v = self.soft_v_eval_step(v, policy, beta, r=r)
            error = np.max(abs(new_v - v))
            v = new_v
            if logging and it % log_steps == 0:
                print('step: ', it, ', error: ', error)
        return v

    def approx_soft_q_eval(self, q: np.ndarray, policy: np.ndarray, beta: float, max_iters: float = 50,
                           tol: float = 1e-9, log_steps: int = 5, logging: bool = True,
                           r: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate optimal soft Q-value function via soft-Q iteration.

        :param q: Initial Q values.
        :param beta: Regularization parameter > 0.
        :param max_iters: Max number of iterations.
        :param tol: Error tolerance for stopping.
        :param log_steps: Pause time for logging.
        :return: q: Optimal Q values.
        """

        it = 0
        error = float("inf")
        while it < max_iters and error > tol:
            it += 1
            new_q = self.soft_q_eval_step(q, policy, beta, r=r)
            error = np.max(abs(new_q - q))
            q = new_q
            if logging and it % log_steps == 0:
                print('step: ', it, ', error: ', error)
        return q

    def approx_vector_cost_eval(self, v: np.ndarray, c: np.ndarray, policy: np.ndarray, max_iters: float = 50,
                          tol: float = 1e-9, log_steps: int = 5, logging: bool = True) -> np.ndarray:
        """
        Evaluate values of policy by iterative Bellman updates.

        :param v: Initial constraint_values, shape (n, n_dims).
        :param policy, shape (n,m).
        :param c, shape (n,m,n_dims)
        :param max_iters: Max number of iterations.
        :param tol: Error tolerance for stopping.
        :param log_steps: Pause time for logging.
        :return: v, shape (n, n_dims).
        """

        it = 0
        error = float("inf")
        while it < max_iters and error > tol:
            it += 1
            new_constraint_values = self.vector_cost_eval_step(v, c, policy)
            error = np.max(abs(new_constraint_values - v))
            v = new_constraint_values
            if logging and it % log_steps == 0:
                print('step: ', it, ', error: ', error)
        return v

    # value and q evaluation via solving the linear equations
    def v_eval(self, policy: np.ndarray, r: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Evaluate value function of policy by solving the linear equations.

        :policy: Policy, shape (n,m).
        :return: v: Values, shape (n,).
        """
        r = self.r if r is None else r
        P_policy = einsum(self.P, policy, 's a s_next, s a -> s s_next')
        r_policy = np.sum(policy * r, axis=1)
        return np.linalg.solve(np.eye(self.n) - self.gamma * P_policy, r_policy)

    # value and q evaluation via solving the linear equation
    def q_eval(self, policy: np.ndarray, r: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Evaluate Q-value function of policy by solving the linear equations.

        :policy: Policy, shape (n,m).
        :return: q: Q-values, shape (n,m).
        """
        r = self.r if r is None else r
        P_policy = rearrange(einsum(self.P, policy, 's a s_next, s_next a_next -> s a s_next a_next'), 's a s_next a_next -> (s a) (s_next a_next)')
        r_f = rearrange(r, 's a -> (s a)')
        return rearrange(np.linalg.solve(np.eye(self.n * self.m) - self.gamma * P_policy, r_f),
                         '(s a) -> s a', s=self.n, a=self.m)

    def soft_v_eval(self, policy: np.ndarray, beta: float, r: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Evaluate soft value function of policy by solving the linear equations.

        :policy: Policy, shape (n,m).
        :param beta: Regularization parameter > 0.
        :return: v: values, shape (n,).
        """

        r = self.r if r is None else r
        P_policy = einsum(self.P, policy, 's a s_next, s a -> s s_next')
        r_policy = np.sum(policy * r, axis=1) + beta * entropy(policy, axis=1)
        return np.linalg.solve(np.eye(self.n) - self.gamma * P_policy, r_policy)

    def soft_q_eval(self, policy: np.ndarray, beta: float, r: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Evaluate soft Q-value function of policy by solving the linear equations.

        :policy: Policy, shape (n,m).
        :param beta: Regularization parameter > 0.
        :return: q: Q-values, shape (n,m).
        """
        r = self.r if r is None else r
        P_policy = rearrange(einsum(self.P, policy, 's a s_next, s_next a_next -> s a s_next a_next'), 's a s_next a_next -> (s a) (s_next a_next)')
        r_regularized = r + beta * self.gamma * rearrange(einsum(self.P, entropy(policy, axis=1), 's a s_next, s_next-> s a'), 's a->(s a)')
        return rearrange(np.linalg.solve(np.eye(self.n * self.m) - self.gamma * P_policy, r_regularized),
                    '(s a) -> s a', s=self.n, a=self.m)

    # greedy policies / policy improvement steps
    def v_greedy(self, v: np.ndarray, r: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get greedy policy from values.

        :param v: values, shape (n,).
        :return: Greedy policy, shape (n,m).
        """
        r = self.r if r is None else r
        q = r + self.gamma * einsum(self.P, v, 's a s_next, s_next -> s a')
        p_mask = q == np.expand_dims(q.max(axis=1), axis=1)
        return p_mask / np.expand_dims(np.sum(p_mask, axis=1), axis=1)

    def q_greedy(self, q: np.ndarray) -> np.ndarray:
        """
        Get greedy policy from q.

        :param q: Q-values, shape (n,m).
        :return: Greedy policy, shape (n,m).
        """
        p_mask = q == np.expand_dims(q.max(axis=1), axis=1)
        return p_mask / np.expand_dims(np.sum(p_mask, axis=1), axis=1)

    def soft_v_greedy(self, v: np.ndarray, beta: float, r: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get soft greedy policy from soft values.

        :param v: values, shape (n,).
        :param beta: Regularization parameter.
        :return: Greedy policy, shape (n,m).
        """
        r = self.r if r is None else r
        q = r + self.gamma * einsum(self.P, v, 's a s_next, s_next -> s a')
        return softmax(q / beta, axis=1)

    def soft_q_greedy(self, q: np.ndarray, beta: float) -> np.ndarray:
        """
        Get soft greedy policy from soft q values.

        :param q: Q-values, shape (n,m).
        :param beta: Regularization parameter.
        :return: Greedy policy, shape (n,m).
        """
        return softmax(q / beta, axis=1)

    # value and q-value iteration
    def v_it(self, v: np.ndarray, max_iters: float = 50,
                         tol: float = 1e-5, log_steps: int = 5, logging: bool = 'True', r: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate optimal value function via value iteration.

        :param v: Initial values, shape (n,).
        :param max_iters: Max number of iterations.
        :param tol: Error tolerance for stopping.
        :return: v: Optimal v, shape (n,).
        """

        it = 0
        error = float("inf")
        if logging:
            print("start value iteration:")
        while it < max_iters and error > tol:
            it += 1
            new_v = self.v_it_step(v, r=r)
            error = np.max(abs(new_v - v))
            v = new_v
            if logging and it % log_steps == 0:
                print('step: ', it, ', error: ', error)
        return v

    def q_it(self, q: np.ndarray, max_iters: float = 50,
                         tol: float = 1e-5, log_steps: int = 5, logging: bool = 'True', r: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate optimal Q-value function via Q iteration.

        :param q: Initial Q v, shape (n,m).
        :param max_iters: Max number of iterations.
        :param tol: Error tolerance for stopping.
        :return: q: Optimal Q v, shape (n,m).
        """

        it = 0
        error = float("inf")
        if logging:
            print("start q iteration:")
        while it < max_iters and error > tol:
            it += 1
            new_q = self.q_it_step(q, r=r)
            error = np.max(abs(new_q - q))
            q = new_q
            if logging and it % log_steps == 0:
                print('step: ', it, ', error: ', error)
        return q

    def soft_v_it(self, v: np.ndarray, beta: float, max_iters: float = 50,
                         tol: float = 1e-5, log_steps: int = 5, logging: bool = 'True', r: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate optimal soft value function via soft value iteration.

        :param v: Initial values, shape (n,).
        :param beta: Regularization parameter > 0.
        :param max_iters: Max number of iterations.
        :param tol: Error tolerance for stopping.
        :return: v: Optimal values, shape (n,).
        """

        it = 0
        error = float("inf")
        if logging:
            print("start soft value iteration:")
        while it < max_iters and error > tol:
            it += 1
            new_v = self.soft_v_it_step(v, beta, r=r)
            error = np.max(abs(new_v - v))
            v = new_v
            if logging and it % log_steps == 0:
                print('step: ', it, ', error: ', error)
        return v

    def soft_q_it(self, q: np.ndarray, beta: float, max_iters: float = 50,
                         tol: float = 1e-5, log_steps: int = 5, logging: bool = 'True', r: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate optimal soft Q-value function via soft-Q iteration.

        :param q: Initial Q v, shape (n,m).
        :param beta: Regularization parameter > 0.
        :param max_iters: Max number of iterations.
        :param tol: Error tolerance for stopping.
        :return: q: Optimal Q v, shape (n,m).
        """

        it = 0
        error = float("inf")
        if logging:
            print("start soft q iteration:")
        while it < max_iters and error > tol:
            it += 1
            new_q = self.soft_q_it_step(q, beta, r=r)
            error = np.max(abs(new_q - q))
            q = new_q
            if logging and it % log_steps == 0:
                print('step: ', it, ', error: ', error)
        return q

    # policy and q-policy iteration
    def policy_it(self, v: np.ndarray, mode: str = 'exact', n_eval_steps: int = 10, eval_tol: float = 1e-9,
                  max_iters: float = 50, tol: float = 1e-5, log_steps: int = 5, logging: bool = 'True', r: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate optimal value function via policy iteration.

        :param v: Initial values, shape (n,).
        :param mode: String in {'exact', 'approx'} indicating policy evaluation mode.
        :param n_eval_steps: Number of Bellman updates for policy evaluation.
        :param eval_tol: Tolerance for stopping Bellman updates.
        :param max_iters: Max number of iterations.
        :param tol: Error tolerance for stopping.
        :param log_steps: Logging interval.
        :param logging: Whether to print outputs or not.
        :param r: Optional reward different from self.r.
        :return: v: Optimal values, shape (n,).
        """

        if mode == 'exact':
            it = 0
            error = float("inf")
            if logging:
                print("start exact policy iteration:")
            while it < max_iters and error > tol:
                it += 1
                policy = self.v_greedy(v, r=r)
                new_v = self.v_eval(policy, r=r)
                error = np.max(abs(new_v - v))
                v = new_v
                if logging and it % log_steps == 0:
                    print('step: ', it, ', error: ', error)
            return v
        else:
            it = 0
            error = float("inf")
            if logging:
                print("start approximative policy iteration:")
            while it < max_iters and error > tol:
                it += 1
                policy = self.v_greedy(v, r=r)
                new_v = self.approx_v_eval(v, policy, max_iters=n_eval_steps, tol=eval_tol, r=r, logging=False)
                error = np.max(abs(new_v - v))
                v = new_v
                if logging and it % log_steps == 0:
                    print('step: ', it, ', error: ', error)
            return v

    def q_policy_it(self, q: np.ndarray, mode: str = 'exact', n_eval_steps: int = 10, eval_tol: float = 1e-9,
                  max_iters: float = 50, tol: float = 1e-5, logging: bool = 'True', log_steps: int = 5, r: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate optimal q value function via q policy iteration.

        :param q: Initial q values, shape (n, m).
        :param mode: String in {'exact', 'approx'} indicating policy evaluation mode.
        :param n_eval_steps: Number of Bellman updates for policy evaluation.
        :param eval_tol: Tolerance for stopping Bellman updates.
        :param max_iters: Max number of iterations.
        :param tol: Error tolerance for stopping.
        :param log_steps: Logging interval.
        :param logging: Whether to print outputs or not.
        :param r: Optional reward different from self.r.
        :return: q: Optimal q values, shape (n, m).
        """

        if mode == 'exact':
            it = 0
            error = float("inf")
            if logging:
                print("start exact q policy iteration:")
            while it < max_iters and error > tol:
                it += 1
                policy = self.q_greedy(q)
                new_q = self.q_eval(policy, r=r)
                error = np.max(abs(new_q - q))
                q = new_q
                if logging and it % log_steps == 0:
                    print('step: ', it, ', error: ', error)
            return q
        else:
            it = 0
            error = float("inf")
            if logging:
                print("start approximative q policy iteration:")
            while it < max_iters and error > tol:
                it += 1
                policy = self.q_greedy(q)
                new_q = self.approx_q_eval(q, policy, max_iters=n_eval_steps, tol=eval_tol, r=r, logging=False)
                error = np.max(abs(new_q - q))
                q = new_q
                if logging and it % log_steps == 0:
                    print('step: ', it, ', error: ', error)
            return q

    def soft_policy_it(self, v: np.ndarray, beta: float, mode: str = 'exact', n_eval_steps: int = 10, eval_tol: float = 1e-9,
                  max_iters: float = 50, tol: float = 1e-5, log_steps: int = 5, logging: bool = 'True', r: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate optimal soft value function via soft policy iteration.

        :param v: Initial soft v, shape (n,).
        :param beta: Regularization parameter > 0.
        :param mode: String in {'exact', 'approx'} indicating policy evaluation mode.
        :param n_eval_steps: Number of Bellman updates for policy evaluation.
        :param eval_tol: Tolerance for stopping Bellman updates.
        :param max_iters: Max number of iterations.
        :param tol: Error tolerance for stopping.
        :param log_steps: Logging interval.
        :param logging: Whether to print outputs or not.
        :param r: Optional reward different from self.r.
        :return: v: Optimal soft values, shape (n,).
        """

        if mode == 'exact':
            it = 0
            error = float("inf")
            if logging:
                print("start exact soft policy iteration:")
            while it < max_iters and error > tol:
                it += 1
                policy = self.soft_v_greedy(v, beta, r=r)
                new_v = self.soft_v_eval(policy, beta, r=r)
                error = np.max(abs(new_v - v))
                v = new_v
                if logging and it % log_steps == 0:
                    print('step: ', it, ', error: ', error)
            return v
        else:
            it = 0
            error = float("inf")
            if logging:
                print("start approximative soft policy iteration:")
            while it < max_iters and error > tol:
                it += 1
                policy = self.soft_v_greedy(v, beta, r=r)
                new_v = self.approx_soft_v_eval(v, policy, beta, max_iters=n_eval_steps, tol=eval_tol, r=r, logging=False)
                error = np.max(abs(new_v - v))
                v = new_v
                if logging and it % log_steps == 0:
                    print('step: ', it, ', error: ', error)
            return v

    def soft_q_policy_it(self, q: np.ndarray, beta: float, mode: str = 'exact', n_eval_steps: int = 10, eval_tol: float = 1e-9,
                  max_iters: float = 50, tol: float = 1e-5, log_steps: int = 5, logging: bool = 'True', r: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate optimal soft q value function via soft q policy iteration.

        :param q: Initial soft q values, shape (n, m).
        :param beta: Regularization parameter > 0.
        :param mode: String in {'exact', 'approx'} indicating policy evaluation mode.
        :param n_eval_steps: Number of Bellman updates for policy evaluation.
        :param eval_tol: Tolerance for stopping Bellman updates.
        :param max_iters: Max number of iterations.
        :param tol: Error tolerance for stopping.
        :param log_steps: Logging interval.
        :param logging: Whether to print outputs or not.
        :param r: Optional reward different from self.r.
        :return: q: Optimal soft q values, shape (n, m).
        """

        if mode == 'exact':
            it = 0
            error = float("inf")
            if logging:
                print("start exact soft q policy iteration:")
            while it < max_iters and error > tol:
                it += 1
                policy = self.soft_q_greedy(q, beta)
                new_q = self.soft_q_eval(policy, beta, r=r)
                error = np.max(abs(new_q - q))
                q = new_q
                if logging and it % log_steps == 0:
                    print('step: ', it, ', error: ', error)
            return q
        else:
            it = 0
            error = float("inf")
            if logging:
                print("start approximative soft q policy iteration:")
            while it < max_iters and error > tol:
                it += 1
                policy = self.soft_q_greedy(q, beta)
                new_q = self.approx_soft_q_eval(q, policy, beta, max_iters=n_eval_steps, tol=eval_tol, r=r, logging=False)
                error = np.max(abs(new_q - q))
                q = new_q
                if logging and it % log_steps == 0:
                    print('step: ', it, ', error: ', error)
            return q

    # Evaluate occupancy measure of some policy
    def policy2stateocc(self, policy: np.ndarray, nu0: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Evaluate state occupancy measure corresponding to policy.

        :param: Policy, shape (n,m).
        :param: Mu, initial state distribution, shape (n,).
        :return: State occupancy measure, shape (n,).
        """

        if nu0 is None:
            nu0 = self.nu0
        P_policy = einsum(self.P, policy, 's a s_next, s a -> s s_next')
        occ = (1-self.gamma) * np.linalg.solve(np.eye(self.n) - self.gamma * P_policy.T, nu0)
        return occ / np.sum(occ)

    # basic MDP tools
    def policy2stateactionocc(self, policy: np.ndarray, nu0: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Evaluate state-action occupancy measure corresponding to policy.

        :policy: Policy, shape (n,m).
        :return: State occupancy measure, shape (n,m).
        """
        if nu0 is None:
            nu0 = self.nu0
        occ = einsum(self.policy2stateocc(policy, nu0), policy, 's, s a -> s a')
        return occ / np.sum(occ)

    def occ2policy(self, occ: np.ndarray) -> np.ndarray:

        z = np.where(np.sum(occ, axis=1, keepdims=True) > 1e-10, occ, 1.0)
        return z / np.sum(z, axis=1, keepdims=True)

    def tens2vec(self, X: np.ndarray):
        if len(X.shape) == 3:
            return rearrange(X, 's a k -> (a s) k')
        elif len(X.shape) == 2:
            return rearrange(X, 's a -> (a s)')
        else:
            raise ValueError('X should have shape (s, a, k) or (s, a)')

    def vec2tens(self, X: np.ndarray):
        if len(X.shape) == 2:
            return rearrange(X, '(a s) k -> s a k', a=self.m)
        elif len(X.shape) == 1:
            return rearrange(X, '(a s) -> s a', a=self.m)
        else:
            raise ValueError('X should be vector or matrix')

    def E_matrix(self):
        E = repeat(np.eye(self.n), 's s_next -> (a s) s_next', a=self.m)
        return E

    def P_matrix(self):
        P = rearrange(self.P, 's a s_next -> (a s) s_next')
        return P

    def A_matrix(self):
        E = repeat(np.eye(self.n), 's s_next -> (a s) s_next', a=self.m)
        P = rearrange(self.P, 's a s_next -> (a s) s_next')
        return E - self.gamma * P

    def A_tensor(self):
        return rearrange(self.A_matrix(), '(a s) s_next -> s a s_next', a=self.m)


    # solve unregularized cmdp with LP solver
    def lp_solve(self):
        E = self.E_matrix()
        P = self.P_matrix()
        r = self.tens2vec(self.r)
        sol = linprog(-r / (1 - self.gamma), A_eq=E.T - self.gamma * P.T, b_eq=(1 - self.gamma) * self.nu0, bounds=(0, None))
        # print(sol.message)
        return self.vec2tens(sol.x), sol
