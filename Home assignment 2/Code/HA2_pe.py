import csv
import numpy as np
from collections import defaultdict
import os
import matplotlib.pyplot as plt

class riverswim():

	def __init__(self, nS):
		self.nS = nS
		self.nA = 2

		# We build the transitions matrix P, and its associated support lists.
		self.P = np.zeros((nS, 2, nS))
		for s in range(nS):
			if s == 0:
				self.P[s, 0, s] = 1
				self.P[s, 1, s] = 0.6
				self.P[s, 1, s + 1] = 0.4
			elif s == nS - 1:
				self.P[s, 0, 0] = 1
				self.P[s, 1, 0] = 1
			else:
				self.P[s, 0, s - 1] = 1
				self.P[s, 1, s] = 0.55
				self.P[s, 1, s + 1] = 0.4
				self.P[s, 1, s - 1] = 0.05
		
		# We build the reward matrix R.
		self.R = np.zeros((nS, 2))
		self.R[0, 0] = 0.05
		self.R[nS - 1, 0] = 1
		self.R[nS - 1, 1] = 1

		# We (arbitrarily) set the initial state in the leftmost position.
		self.s = 0

###############################################################################
# 1. Reading the dataset and grouping by episodes
###############################################################################

def load_episodes_from_csv(csv_path, terminal_state=None):
    """
    Reads dataset from csv_path. Have lines like:
       state,action,reward,next state
    and returns a list of episodes, where each episode is a list of (s,a,r,s') tuples
    """
    episodes = []
    current_episode = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)  # columns: state, action, reward, next state
        for row in reader:
            s = int(row['state'])
            a = int(row['action'])
            r = float(row['reward'])
            s_next = int(row['next state'])
            
            current_episode.append((s, a, r, s_next))
            
            # If a terminal state is reached, end the episode
            if terminal_state is not None and s_next == terminal_state:
                episodes.append(current_episode)
                current_episode = []
            
    # Append any remaining transitions as an episode.
    if current_episode:
        episodes.append(current_episode)
    
    return episodes

###############################################################################
# 2. Definitions of behavior policy pi_b and target policy pi
###############################################################################

def pi_b(s, a):
    """
    Behavior policy from which the dataset was generated:
    Probability that the dataset's logging policy chooses action `a` in state `s`
    """
    # Behavior was 65% 'action=1' and 35% 'action=0'
    if a==0:
        return 0.35
    elif a==1:
        return 0.65
    else:
        return 0.0

def pi(s, a):
    """
    Target policy's probability of choosing action `a` in state `s`
    We define a simple "go right" policy
    """
    # Deterministic at 'action=1'.
    return 1.0 if (a==1) else 0.0

###############################################################################
# 3. Model-Based OPE
###############################################################################

def MB_OPE(episodes, gamma):
    """
    Model-Based OPE:
     1) Estimate P_hat(s'|s,a) and R_hat(s,a) from data
     2) For each state s, r^pi(s)= sum_a [ pi(a|s)* R_hat(s,a) ]
                  P^pi(s'|s)= sum_a [ pi(a|s)* P_hat(s'|s,a) ]
     3) Solve  (I - gamma P^pi) V = r^pi  =>  V = (I - gamma P^pi)^{-1} r^pi
    Returns: MB-based estimate of V^pi(s) for all states s as a vector.
    """
    
    # figure out how large state space can be
    max_s = 0
    max_a = 0
    for ep in episodes:
        for (s,a,r,s_next) in ep:
            max_s = max(max_s, s, s_next)
            max_a = max(max_a, a)
    nS = max_s + 1
    nA = max_a + 1
    
    N_sa = np.zeros((nS,nA))  # how many times (s,a) was visited
    sumR_sa = np.zeros((nS,nA))
    N_sasprime = np.zeros((nS,nA,nS))
    
    # Populate counts
    for ep in episodes:
        for (s,a,r,s_next) in ep:
            N_sa[s,a] += 1
            sumR_sa[s,a] += r
            N_sasprime[s,a,s_next] += 1
    
    # Build R_hat and P_hat
    R_hat = np.zeros((nS,nA))
    P_hat = np.zeros((nS,nA,nS))
    
    for s in range(nS):
        for a in range(nA):
            if N_sa[s,a] > 0:
                R_hat[s,a] = sumR_sa[s,a]/ N_sa[s,a]
                P_hat[s,a,:] = N_sasprime[s,a,:]/ N_sa[s,a]
            else:
                # If never visited, default to 0 reward and uniform transitions
                P_hat[s,a,:] = 1.0/nS
    
    # Now form P^pi and r^pi
    Ppi = np.zeros((nS,nS))
    rpi = np.zeros(nS)
    
    for s in range(nS):
        for a in range(nA):
            rpi[s] += pi(s,a)* R_hat[s,a]
        for s_next in range(nS):
            # sum_a pi(a|s)* P_hat(s'|s,a)
            val = 0.0
            for a in range(nA):
                val += pi(s,a)* P_hat[s,a,s_next]
            Ppi[s,s_next] = val
    
    # Solve linear system:  (I - gamma P^pi) V = r^pi
    A = np.eye(nS) - gamma*Ppi
    b = rpi
    V_est = np.linalg.solve(A, b)
    
    return V_est

###############################################################################
# 4. Off-Policy IS Estimators
###############################################################################

def V_pi_IS(episodes, gamma):
    """
    Basic (Trajectory-level) Importance Sampling for V^pi(s_init).
    We assume all episodes start in the same state s_init,
    and define
        Vhat_IS = 1/n sum_{i=1}^n [ rho_{1:Ti}^{(i)} * sum_{t=1}^{Ti} gamma^{t-1} r_t^{(i)} ]
    with  rho_{1:T}^{(i)} = product_{t=1}^T [ pi(a_t|s_t)/ pi_b(a_t|s_t) ].
    """
    G = []
    for ep in episodes:
        # compute entire-trajectory ratio and sum of discounted rewards
        rho_1_T = 1.0
        discounted_return = 0.0
        for t,(s,a,r,s_next) in enumerate(ep, start=1):
            # multiply ratio
            rho_1_T *= (pi(s,a) / pi_b(s,a))
            discounted_return += (gamma**(t-1))* r
        G.append(rho_1_T * discounted_return)
    return np.mean(G)

def V_pi_wIS(episodes, gamma):
    """
    Weighted Importance Sampling:
      Vhat_wIS = [ sum_{i=1}^n rho_{1:Ti}^{(i)} * sum_{t=1}^{Ti} gamma^{t-1} r_t^{(i)} ]
                  / sum_{i=1}^n rho_{1:Ti}^{(i)}.
    This is consistent, slightly biased, with lower variance.
    """
    numerator = 0.0
    denominator = 0.0
    for ep in episodes:
        rho_1_T = 1.0
        discounted_return = 0.0
        for t,(s,a,r,s_next) in enumerate(ep, start=1):
            rho_1_T *= (pi(s,a)/ pi_b(s,a))
            discounted_return += (gamma**(t-1))* r
        numerator += rho_1_T* discounted_return
        denominator+= rho_1_T
    if denominator == 0.0:
        return 0.0
    return numerator/ denominator

def V_pi_PDIS(episodes, gamma):
    """
    Per-Decision IS (PDIS):
      Vhat_PDIS = 1/n sum_{i=1}^n sum_{t=1}^{T_i} [ (prod_{k=1}^t pi(a_k|s_k)/ pi_b(a_k|s_k)) * gamma^{t-1} * r_t ].
    """
    n = len(episodes)
    accum = 0.0
    
    for ep in episodes:
        partial_sum = 0.0
        rho_1_t = 1.0
        for t,(s,a,r,s_next) in enumerate(ep, start=1):
            rho_1_t *= pi(s,a)/ pi_b(s,a)
            partial_sum += rho_1_t * (gamma**(t-1)) * r
        accum += partial_sum
    return accum / n

def plot_errors(episodes, gamma, V_true):
    """
    Computes the running OPE estimates as a function of the number of episodes,
    compares them to the 'true' V^pi (computed via MB-OPE on the full dataset),
    and plots the absolute error |V^pi(s_init) - estimate| for each method.
    """

    s_init = episodes[0][0][0] # initial state from the first episode
    true_value = V_true[s_init]
    # Lists to store errors and number of episodes used.
    errors_IS, errors_wIS, errors_PDIS, errors_MB = [], [], [], []
    episode_counts = []

    # For each prefix of the dataset, compute the running estimates
    for i in range(1, len(episodes)+1):
        current_eps = episodes[:i]
        v_is   = V_pi_IS(current_eps, gamma)
        v_wis  = V_pi_wIS(current_eps, gamma)
        v_pdis = V_pi_PDIS(current_eps, gamma)
        v_mb   = MB_OPE(current_eps, gamma)[s_init]
        
        errors_IS.append(abs(true_value - v_is))
        errors_wIS.append(abs(true_value - v_wis))
        errors_PDIS.append(abs(true_value - v_pdis))
        errors_MB.append(abs(true_value - v_mb))
        episode_counts.append(i)

    # Plot the error curves
    plt.figure(figsize=(10,6))
    plt.plot(episode_counts, errors_IS, label='IS Error', marker='o')
    plt.plot(episode_counts, errors_wIS, label='wIS Error', marker='s')
    plt.plot(episode_counts, errors_PDIS, label='PDIS Error', marker='^')
    plt.plot(episode_counts, errors_MB, label='MB-OPE Error', marker='d')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Absolute Error |V^π(s_init) - Estimate|')
    plt.title('OPE Estimate Errors vs. True V^π(s_init)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__=="__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    csv_file = "Datasets/dataset0.csv"
    # Load episodes from CSV
    episodes = load_episodes_from_csv(csv_file, terminal_state=5)
    gamma = 0.96  # discount factor, example
    s_init = 0
    
    ###########################################################################
    # i) Estimate V^pi(s_init) using: MB-OPE, IS, wIS, and PDIS
    ###########################################################################

    # Model-Based OPE
    mb_ope_est = MB_OPE(episodes, gamma)
    print(f"MB-OPE estimate of V^pi(s_init): {mb_ope_est[s_init]:.4f}")
    # IS, wIS, PDIS
    v_is   = V_pi_IS(episodes, gamma)
    v_wis  = V_pi_wIS(episodes, gamma)
    v_pdis = V_pi_PDIS(episodes, gamma)
    print(f"IS estimate of V^pi(s_init): {v_is:.4f}")
    print(f"wIS estimate of V^pi(s_init): {v_wis:.4f}")
    print(f"PDIS estimate of V^pi(s_init): {v_pdis:.4f}")

    ###########################################################################
    # ii) For each method, plot the error 
    ###########################################################################

    env = riverswim(6)
    # Need to solve (I - gamma*P^pi)v = r^pi
    # First build P^pi and r^pi for our policy
    P_pi = np.zeros((env.nS, env.nS))
    r_pi = np.zeros(env.nS)
    # Build P^pi and r^pi matrices based on our policy
    for s in range(env.nS):
        # r^pi is weighted average of rewards for each action
        r_pi[s] = (pi(s,0)*env.R[s,0]) + (pi(s,1)*env.R[s,1])
        
        # P^pi combines transition probs weighted by policy probs
        for s_next in range(env.nS):
            P_pi[s, s_next] = (pi(s,0)*env.P[s,0,s_next]
                            + pi(s,1)*env.P[s,1,s_next])

    # Solve the system (I - gamma*P^pi)v = r^pi using numpy's solver because it's more stable
    I = np.eye(env.nS)
    A = I - gamma * P_pi
    b = r_pi

    V_true = np.linalg.solve(A, b)

    print(f"Exact solution of V^pi(s_init) using linear system: : {V_true[s_init]:.4f}")

    # Plot error curves
    plot_errors(episodes, gamma, V_true)

    ###########################################################################
    # iii) Consider 9 additional datasets and report empirical variance
    ###########################################################################

    methods = ["MB-OPE", "IS", "wIS", "PDIS"]
    # Dictionaries to store the absolute errors and estimates for each method
    abs_errors = {m: [] for m in methods}
    estimates_dict = {m: [] for m in methods}

    # Loop over datasets dataset0.csv to dataset9.csv
    for i in range(10):
        csv_file_i = f"Datasets/dataset{i}.csv"
        eps_i = load_episodes_from_csv(csv_file_i, terminal_state=5)
        
        # Compute the estimates for state s_init using each method
        mb_est = MB_OPE(eps_i, gamma)[s_init]
        is_est = V_pi_IS(eps_i, gamma)
        wis_est = V_pi_wIS(eps_i, gamma)
        pdis_est = V_pi_PDIS(eps_i, gamma)
        
        estimates_dict["MB-OPE"].append(mb_est)
        estimates_dict["IS"].append(is_est)
        estimates_dict["wIS"].append(wis_est)
        estimates_dict["PDIS"].append(pdis_est)
        
        # Compute the absolute error relative to the exact V^pi(s_init)
        abs_errors["MB-OPE"].append(abs(mb_est - V_true[s_init]))
        abs_errors["IS"].append(abs(is_est - V_true[s_init]))
        abs_errors["wIS"].append(abs(wis_est - V_true[s_init]))
        abs_errors["PDIS"].append(abs(pdis_est - V_true[s_init]))

    print("\nEstimates for V^pi(s_init) across datasets:")
    for m in methods:
        var_est = np.var(estimates_dict[m])
        print(f"{m}: {np.array(estimates_dict[m]).round(4)}, Variance = {var_est:.4f}")

    ###########################################################################
    # iv) Compare in terms of empirical error and variance
    ###########################################################################

    # Print the mean absolute error and its variance for each method
    print("\nEmpirical comparison across 10 datasets:")
    for m in methods:
        mean_error = np.mean(abs_errors[m])
        var_error = np.var(abs_errors[m])  # sample variance
        print(f"{m}: Mean Absolute Error = {mean_error:.4f}, Variance = {var_error:.4f}")