import csv
import numpy as np
from collections import defaultdict
import os

###############################################################################
# 1. Reading the dataset and grouping by episodes
###############################################################################

def load_episodes_from_csv(csv_path):
    """
    Reads dataset from csv_path, assumed to have lines like:
       state,action,reward,next_state
    and returns a list of episodes, where each episode is a list of (s,a,r,s') tuples.
    We assume that 'next_state' might loop or eventually reach a terminal.
    In a continuing task, you might artificially chunk by restarts if you wish.
    """
    episodes = []
    current_episode = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)  # columns: state, action, reward, next_state
        prev_next_state = None
        for row in reader:
            s     = int(row['state'])
            a     = int(row['action'])
            r     = float(row['reward'])
            s_next= int(row['next state'])
            
            current_episode.append((s, a, r, s_next))
            
            # Here, as an example, we split episodes whenever s_next < s 
            # or if you'd prefer a known terminal condition, 
            # you can replace it with (s_next == terminal_state) etc.
            # For a simpler demonstration, let's say we'll just continue
            # one long trajectory. If your data truly has multiple episodes,
            # insert your own splitting logic here.
            
        # in the simplest case we treat everything as a single "episode"
        episodes.append(current_episode)
    
    return episodes

###############################################################################
# 2. Example definitions for behavior policy pi_b and target policy pi
###############################################################################

def pi_b(s, a):
    """
    Behavior policy from which the dataset was generated:
    Probability that the dataset's logging policy chooses action `a` in state `s`.
    Must match how data was actually logged (if known).
    For the coverage assumption pi(a|s) > 0 => pi_b(a|s) > 0, we need pi_b>0 whenever pi>0.
    
    Here we just put a placeholder. In a real scenario, you must provide
    the correct probability from the logging policy.
    """
    # Example: Suppose behavior was 65% 'action=1' if s=0, else uniform. TOTALLY ARBITRARY
    # Adjust as needed:
    nA = 2  # or whatever your action set size is
    if s==0:
        if a==1:
            return 0.65
        else:
            return 0.35
    else:
        # uniform among 0..(nA-1)
        return 1.0/nA

def pi(s, a):
    """
    Target policy's probability of choosing action `a` in state `s`.
    For example, we can define a simple "go right" policy or anything relevant.
    """
    nA = 2
    # Suppose the target policy always does action=1 (prob=1),
    # i.e. deterministic at 'action=1'.
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
    # First gather counts for transitions and sum of rewards
    # Let's assume we know the state space S and action set A from the data
    # We'll do a pass to figure out max states or so:
    
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
                # If never visited, default to 0 reward and uniform transitions, e.g.
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
        Vhat_IS = 1/n \sum_{i=1}^n [ rho_{1:Ti}^{(i)} * sum_{t=1}^{Ti} gamma^{t-1} r_t^{(i)} ]
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
    This is consistent, slightly biased, but can have lower variance.
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

###############################################################################
# Example usage
###############################################################################

if __name__=="__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    csv_file = "Datasets/dataset0.csv"
    # 1) Load episodes from CSV
    episodes = load_episodes_from_csv(csv_file)
    gamma = 0.96  # discount factor, example
    
    # 2) Model-Based OPE
    mb_ope_est = MB_OPE(episodes, gamma)
    print("MB-OPE estimate of V^pi(s) for all states:\n", mb_ope_est)
    
    # If we only care about s_init=0 (assuming all eps start in 0):
    # you can check mb_ope_est[0] for the MB-OPE of V^pi(0).
    
    # 3) IS, wIS, PDIS
    v_is   = V_pi_IS(episodes, gamma)
    v_wis  = V_pi_wIS(episodes, gamma)
    v_pdis = V_pi_PDIS(episodes, gamma)
    print(f"IS estimate of V^pi(s_init): {v_is:.4f}")
    print(f"wIS estimate of V^pi(s_init): {v_wis:.4f}")
    print(f"PDIS estimate of V^pi(s_init): {v_pdis:.4f}")
