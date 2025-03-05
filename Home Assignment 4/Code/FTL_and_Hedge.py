import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Exercise 5.8: Empirical Comparison of FTL and Hedge
# ------------------------------------------------------------------
# We consider a binary sequence X_1, ..., X_T i.i.d. Bernoulli(p).
# There are two experts:
#    - Expert 0 always predicts 0
#    - Expert 1 always predicts 1
#
# We compare:
#   (1) FTL (Follow The Leader)
#   (2) Hedge with a fixed eta = eta_fixed1
#   (3) Hedge with a fixed eta = eta_fixed2
#   (4) Hedge with an "anytime" schedule #1: eta_t = c1 * sqrt( ln(K)/ t )
#   (5) Hedge with an "anytime" schedule #2: eta_t = c2 * sqrt( ln(K)/ t )
#
# The Hedge algorithm is implemented as in the slides:
#   - Initialize L_0(a) = 0 for each expert a.
#   - For t = 1 to T:
#       p_t(a) = exp( - eta_t * L_{t-1}(a) ) / sum_{b} exp( - eta_t * L_{t-1}(b) )
#       Sample A_t ~ p_t(·)
#       Observe the outcome X_t
#       The loss of expert0 is I{X_t=1}, the loss of expert1 is I{X_t=0}.
#       Update L_t(a) = L_{t-1}(a) + loss_of_expert_a_at_time_t
#
# The 0-1 loss of the algorithm at time t is I{prediction != X_t}.
# We measure pseudo-regret relative to the best single expert in hindsight:
#   best_expert_loss(t) = min( # of 1s in first t steps, # of 0s in first t steps ).
# The code below:
#   - defines these algorithms,
#   - runs them multiple times,
#   - computes average pseudo-regret over the runs,
#   - plots the results.
# ------------------------------------------------------------------

def simulate_sequence(T, p):
    """
    Generate a length-T binary sequence i.i.d. Bernoulli(p).
    """
    return np.random.binomial(n=1, p=p, size=T)

def best_expert_loss(X):
    """
    For 2 experts:
      - Expert0 always predicts 0 => mistakes on #1's
      - Expert1 always predicts 1 => mistakes on #0's
    The best single expert in hindsight is the one with fewer total mistakes 
    over the first t steps. So at time t, best_expert_loss[t] = min(#1's, #0's in X[0..t]).
    """
    T = len(X)
    best_loss = np.zeros(T)
    num_ones = 0
    for t in range(T):
        if X[t] == 1:
            num_ones += 1
        num_zeros = (t+1) - num_ones
        best_loss[t] = min(num_ones, num_zeros)
    return best_loss

def ftl_predict(X):
    """
    Follow The Leader for 2 experts (predict0 or predict1).
    At time t, choose the expert that had the fewest mistakes in the first t-1 rounds.
    We track cumulative loss of the algorithm as well.
    """
    T = len(X)
    cum_loss = np.zeros(T)
    # mistakes of each expert so far
    L0 = 0  # # times X[t] = 1
    L1 = 0  # # times X[t] = 0
    alg_mistakes = 0
    for t in range(T):
        # pick the leader
        if L0 < L1:
            pred = 0
        elif L1 < L0:
            pred = 1
        else:
            # tie => break by picking 0
            pred = 0
        
        # see if we made a mistake
        if pred != X[t]:
            alg_mistakes += 1
        
        # update the experts' mistakes
        if X[t] == 1:
            L0 += 1
        else:
            L1 += 1
        
        cum_loss[t] = alg_mistakes
    return cum_loss

def hedge_predict(X, schedule_type, param):
    """
    Hedge algorithm, as in the slides:

    L_0(a) = 0
    For t=1..T:
       p_t(a) = exp(- eta_t * L_{t-1}(a)) / sum_b exp(- eta_t * L_{t-1}(b))
       A_t ~ p_t(·)  (sample from distribution)
       Observe X[t-1]
       update L_t(a) = L_{t-1}(a) + (loss for expert a at time t)

    We define 2 experts (0 => always predict0, 1 => always predict1).
    The schedule_type can be:
       'fixed' => param is a constant, i.e. eta_t = param
       'anytime' => param is c, i.e. eta_t = c * sqrt( ln(K)/ t )

    We'll do a *stochastic* pick of the action: 
      pick action 1 with probability p_t(1), action 0 with prob p_t(0).

    Return the array of cumulative losses of the algorithm.
    """
    T = len(X)
    K = 2  # 2 experts
    # L_{0}(0) = 0, L_{0}(1)=0
    L = np.zeros(K)  
    cum_loss = np.zeros(T)
    
    alg_mistakes = 0
    for t in range(1, T+1):
        # define eta_t
        if schedule_type == 'fixed':
            eta_t = param
        elif schedule_type == 'anytime':
            # K=2 => ln(K)=ln(2)
            eta_t = param * np.sqrt(np.log(K)/t)
        else:
            raise ValueError("schedule_type must be 'fixed' or 'anytime'")
        
        # compute p_t(0) and p_t(1)
        L_min = np.min(L)
        w = np.exp(-eta_t * (L - L_min))
        w_sum = np.sum(w)
        p = w / w_sum

        # sample action from p
        action = np.random.choice([0,1], p=p)
        
        # check mistake
        # The realized outcome is X[t-1]
        if action != X[t-1]:
            alg_mistakes += 1
        
        # update experts' losses
        # Expert0 is wrong if X[t-1]=1
        # Expert1 is wrong if X[t-1]=0
        loss0 = 1 if (X[t-1] == 1) else 0
        loss1 = 1 if (X[t-1] == 0) else 0
        L[0] += loss0
        L[1] += loss1
        
        cum_loss[t-1] = alg_mistakes
    
    return cum_loss

def run_experiment(T, p, n_runs=10):
    """
    Compare 5 algorithms on i.i.d. Bernoulli(p) data:
       1) FTL
       2) Hedge fixed eta_1
       3) Hedge fixed eta_2
       4) Hedge anytime c_1
       5) Hedge anytime c_2

    Return average cumulative losses, plus average best expert loss => pseudo-regret.
    """
    # define parameter sets
    eta_fixed_1 = 0.2
    eta_fixed_2 = 0.5
    c_anytime_1 = 0.5
    c_anytime_2 = 1.0
    
    alg_names = [
        "FTL",
        f"Hedge_fixed({eta_fixed_1})",
        f"Hedge_fixed({eta_fixed_2})",
        f"Hedge_anytime({c_anytime_1})",
        f"Hedge_anytime({c_anytime_2})"
    ]
    
    sum_cum_loss = {name: np.zeros(T) for name in alg_names}
    sum_best_loss = np.zeros(T)
    
    for _ in range(n_runs):
        # generate data
        X = simulate_sequence(T, p)
        
        # run each algorithm
        cum_loss_ftl = ftl_predict(X)
        cum_loss_hf1 = hedge_predict(X, schedule_type='fixed', param=eta_fixed_1)
        cum_loss_hf2 = hedge_predict(X, schedule_type='fixed', param=eta_fixed_2)
        cum_loss_ha1 = hedge_predict(X, schedule_type='anytime', param=c_anytime_1)
        cum_loss_ha2 = hedge_predict(X, schedule_type='anytime', param=c_anytime_2)
        
        # accumulate
        sum_cum_loss["FTL"] += cum_loss_ftl
        sum_cum_loss[f"Hedge_fixed({eta_fixed_1})"] += cum_loss_hf1
        sum_cum_loss[f"Hedge_fixed({eta_fixed_2})"] += cum_loss_hf2
        sum_cum_loss[f"Hedge_anytime({c_anytime_1})"] += cum_loss_ha1
        sum_cum_loss[f"Hedge_anytime({c_anytime_2})"] += cum_loss_ha2
        
        # best expert
        best_l = best_expert_loss(X)
        sum_best_loss += best_l
    
    # average
    avg_cum_loss = {name: sum_cum_loss[name]/n_runs for name in alg_names}
    avg_best_loss = sum_best_loss / n_runs
    
    # pseudo-regret = (algorithm's cum loss) - (best single expert's cum loss)
    avg_pseudo_regret = {}
    for name in alg_names:
        avg_pseudo_regret[name] = avg_cum_loss[name] - avg_best_loss
    
    return avg_pseudo_regret, alg_names

def main():
    T = 200
    p = 0.7
    n_runs = 10
    
    avg_pseudo_regret, alg_names = run_experiment(T, p, n_runs)
    
    # Plot
    plt.figure(figsize=(8,5))
    for name in alg_names:
        plt.plot(avg_pseudo_regret[name], label=name)
    plt.title(f"Exercise 5.8 - p={p}, T={T}, n_runs={n_runs}")
    plt.xlabel("t")
    plt.ylabel("Pseudo-Regret")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
