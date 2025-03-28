# Written by Hippolyte Bourel
# This code is proposed as a reference solution for various exercises of Home Assignements for the OReL course in 2024.
# This solution is tailored for simplicity of understanding and is in no way optimal, nor the only way to implement the different elements!
import numpy as np

# A simple riverswim implementation with chosen number of state 'nS' chosen in input.
# We arbitrarily chose the action '0' = 'go to the left' thus '1' = 'go to the right'.
# Finally the state '0' is the leftmost, 'nS - 1' is the rightmost.
class riverswim():

	def __init__(self, nS):
		self.nS = nS
		self.nA = 2

		# We build the transitions matrix P, and its associated support lists.
		self.P = np.zeros((nS, 2, nS))
		self.support = [[[] for _ in range(self.nA)] for _ in range(self.nS)]
		for s in range(nS):
			if s == 0:
				self.P[s, 0, s] = 1
				self.P[s, 1, s] = 0.6
				self.P[s, 1, s + 1] = 0.4
				self.support[s][0] += [0]
				self.support[s][1] += [0, 1]
			elif s == nS - 1:
				self.P[s, 0, s - 1] = 1
				self.P[s, 1, s] = 0.6
				self.P[s, 1, s - 1] = 0.4
				self.support[s][0] += [s - 1]
				self.support[s][1] += [s - 1, s]
			else:
				self.P[s, 0, s - 1] = 1
				self.P[s, 1, s] = 0.55
				self.P[s, 1, s + 1] = 0.4
				self.P[s, 1, s - 1] = 0.05
				self.support[s][0] += [s - 1]
				self.support[s][1] += [s - 1, s, s + 1]
		
		# We build the reward matrix R.
		self.R = np.zeros((nS, 2))
		self.R[0, 0] = 0.05
		self.R[nS - 1, 1] = 1

		# We (arbitrarily) set the initial state in the leftmost position.
		self.s = 0

	# To reset the environment in initial settings.
	def reset(self):
		self.s = 0
		return self.s

	# Perform a step in the environment for a given action. Return a couple state, reward (s_t, r_t).
	def step(self, action):
		new_s = np.random.choice(np.arange(self.nS), p=self.P[self.s, action])
		reward = self.R[self.s, action]
		self.s = new_s
		return new_s, reward


"""
Optimized UCB Q-Learning on 5-state RiverSwim for Home Assignment 7, Exercise 4:
"An Empirical Evaluation of UCB Q-learning" (25 points)

Parameters:
  - gamma = 0.92
  - epsilon_eval = 0.13 (to check ε-badness)
  - delta = 0.05
  - T = 2e6 steps per run
  - H = ceil((1/(1-gamma))*log(1/epsilon_eval))
  - Bonus: b(k) = sqrt((H/k) * log(S*A*log(k+1)/delta))

We count ε-bad steps as:
    n(t) = sum_{tau=1}^{t} I{ V^π_tau(s_tau) < V*(s_tau) - epsilon_eval }
where V*(s) is computed via value iteration.

This code uses a caching strategy so that the 5×5 system solve for policy evaluation is only performed once per unique policy.
A tqdm progress bar is added for the outer loop over the 100 simulation runs.

Author: [Your Name]
Date: [Date]
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # progress bar for the outer simulation loop
from HA7_RiverSwim import riverswim  # Provided RiverSwim environment

# ---------------------------------------------------------------------
# 1. Helper Functions
# ---------------------------------------------------------------------

def compute_optimal_values(env, gamma, tol=1e-8):
    """
    Compute the optimal value function V* for the given MDP (env)
    using standard value iteration.
    """
    S = env.nS
    A = env.nA
    V = np.zeros(S)
    while True:
        V_new = np.zeros_like(V)
        for s in range(S):
            Q_vals = []
            for a in range(A):
                Q_vals.append(env.R[s, a] + gamma * np.dot(env.P[s, a, :], V))
            V_new[s] = max(Q_vals)
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new
    return V

def evaluate_policy(policy, env, gamma):
    """
    Given a deterministic policy (array of length S), solve:
         V(s) = R(s, policy[s]) + gamma * sum_{s'} P(s, policy[s], s') * V(s')
    """
    S = env.nS
    P_pi = np.zeros((S, S))
    r_pi = np.zeros(S)
    for s in range(S):
        a = policy[s]
        P_pi[s, :] = env.P[s, a, :]
        r_pi[s] = env.R[s, a]
    I = np.eye(S)
    V_policy = np.linalg.solve(I - gamma * P_pi, r_pi)
    return V_policy

# ---------------------------------------------------------------------
# 2. Main UCB Q-Learning Routine (with caching)
# ---------------------------------------------------------------------

def run_ucb_ql(T, gamma, epsilon_eval, delta, H, V_star, record_interval=1000, seed=None):
    """
    Runs UCB Q-learning for T steps on the 5-state RiverSwim MDP.
    Uses caching for policy evaluation to avoid repeated 5x5 solves.
    
    Returns:
       times: array of recording time steps
       n_values: array of cumulative ε-bad step counts
    """
    if seed is not None:
        np.random.seed(seed)

    env = riverswim(5)
    # Set starting state uniformly at random.
    env.s = np.random.randint(0, env.nS)
    s = env.s

    S = env.nS
    A = env.nA
    R_max = 1.0  # from RiverSwim

    optimistic_init = R_max / (1.0 - gamma)
    Q = np.full((S, A), optimistic_init)
    Q_hat = np.copy(Q)
    N = np.ones((S, A))  # visit counts, starting at 1

    n_bad = 0
    times = []
    n_values = []
    
    # Initialize cache for policy evaluation
    policy_value_cache = {}

    for t in range(1, T + 1):
        # Select action greedily with respect to Q_hat.
        a = np.argmax(Q_hat[s, :])
        new_s, r = env.step(a)
        k = N[s, a]
        alpha_k = (H + 1) / (H + k)
        bonus = np.sqrt((H / k) * np.log(S * A * np.log(k + 1) / delta))
        # Q-update with bonus term.
        Q[s, a] = (1 - alpha_k) * Q[s, a] + alpha_k * (r + bonus + gamma * np.max(Q_hat[new_s, :]))
        Q_hat[s, a] = min(Q_hat[s, a], Q[s, a])
        N[s, a] += 1

        # Retrieve current greedy policy and its value function from cache.
        policy = np.argmax(Q_hat, axis=1)
        policy_tuple = tuple(policy)
        if policy_tuple not in policy_value_cache:
            V_policy = evaluate_policy(policy, env, gamma)
            policy_value_cache[policy_tuple] = V_policy
        else:
            V_policy = policy_value_cache[policy_tuple]
        # Count ε-bad step if value at new_s is below V*(new_s) - epsilon_eval.
        if V_policy[new_s] < V_star[new_s] - epsilon_eval:
            n_bad += 1

        s = new_s

        if t % record_interval == 0:
            times.append(t)
            n_values.append(n_bad)

    return np.array(times), np.array(n_values)

# ---------------------------------------------------------------------
# 3. Main Experiment: Single Run and Averaging Over 100 Runs
# ---------------------------------------------------------------------

def main():
    # Experiment parameters
    T = 2_000_000         # Total time steps per run
    gamma = 0.92
    epsilon_eval = 0.13
    delta = 0.05
    record_interval = 1000
    H = int(np.ceil((1.0 / (1.0 - gamma)) * np.log(1.0 / epsilon_eval)))

    # Compute the optimal value function V* via value iteration.
    env_for_opt = riverswim(5)
    V_star = compute_optimal_values(env_for_opt, gamma)
    print("Optimal value function V*:", V_star)

    # (i) Single run: plot sample path of n(t)
    times_sample, n_sample = run_ucb_ql(
        T=T,
        gamma=gamma,
        epsilon_eval=epsilon_eval,
        delta=delta,
        H=H,
        V_star=V_star,
        record_interval=record_interval,
        seed=42
    )
    plt.figure(figsize=(8, 5))
    plt.plot(times_sample, n_sample, label="Single run")
    plt.xlabel("Time steps")
    plt.ylabel("Cumulative ε-bad steps n(t)")
    plt.title("Sample Path of n(t) for UCB-QL (Single Run)")
    plt.grid(True)
    plt.legend()
    plt.savefig("sample_path_ucbql.png", bbox_inches="tight")
    plt.show()

    # (ii) Average over 100 runs with 95% confidence intervals.
    runs = 100
    all_n = []
    for i in tqdm(range(runs), desc="Running 100 simulations"):
        _, n_arr = run_ucb_ql(
            T=T,
            gamma=gamma,
            epsilon_eval=epsilon_eval,
            delta=delta,
            H=H,
            V_star=V_star,
            record_interval=record_interval,
            seed=i
        )
        all_n.append(n_arr)
    all_n = np.array(all_n)  # shape: (runs, num_record_points)
    mean_n = np.mean(all_n, axis=0)
    std_n = np.std(all_n, axis=0)
    ci_upper = mean_n + 1.96 * std_n / np.sqrt(runs)
    ci_lower = mean_n - 1.96 * std_n / np.sqrt(runs)

    plt.figure(figsize=(8, 5))
    plt.plot(times_sample, mean_n, label="Average n(t) over 100 runs")
    plt.fill_between(times_sample, ci_lower, ci_upper, color="gray", alpha=0.3, label="95% CI")
    plt.xlabel("Time steps")
    plt.ylabel("Cumulative ε-bad steps n(t)")
    plt.title("UCB-QL: Average n(t) over 100 runs (with 95% CI)")
    plt.grid(True)
    plt.legend()
    plt.savefig("average_n_t_ucbql.png", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
