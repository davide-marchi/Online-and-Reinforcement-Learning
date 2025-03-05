import numpy as np
import matplotlib.pyplot as plt

def simulate_sequence(T, p):
    """
    Generate a length-T binary sequence i.i.d. Bernoulli(p).
    """
    return np.random.binomial(n=1, p=p, size=T)

def best_expert_loss(X):
    """
    For 2 experts (always predict 0 or always predict 1):
    best_expert_loss[t] = min(#1's in first t steps, #0's in first t steps).
    """
    T = len(X)
    best_loss = np.zeros(T)
    num_ones = 0
    for t in range(T):
        if X[t] == 1:
            num_ones += 1
        num_zeros = (t + 1) - num_ones
        best_loss[t] = min(num_ones, num_zeros)
    return best_loss

def ftl_predict(X):
    """
    Follow The Leader (FTL) for 2 experts:
     - Expert0 always predicts 0
     - Expert1 always predicts 1
    At time t, pick the expert with fewer total mistakes so far.
    Return the array of cumulative losses of FTL (length T).
    """
    T = len(X)
    cum_loss = np.zeros(T)
    L0, L1 = 0, 0  # mistakes for each expert
    alg_mistakes = 0
    for t in range(T):
        # pick the leader
        if L0 < L1:
            pred = 0
        elif L1 < L0:
            pred = 1
        else:
            pred = 0  # tie => pick 0

        # see if we made a mistake
        if pred != X[t]:
            alg_mistakes += 1
        
        # update experts
        if X[t] == 1:
            L0 += 1
        else:
            L1 += 1

        cum_loss[t] = alg_mistakes
    return cum_loss

def hedge_predict(X, schedule_type, param):
    """
    Hedge algorithm with 2 experts (0 => always predict0, 1 => always predict1),
    as in the class slides:

      Initialization: L_0(a) = 0 for each expert a
      for t=1..T:
         if schedule_type='fixed':
             eta_t = param
         else if schedule_type='anytime':
             eta_t = param * sqrt( ln(2) / t )
         w[a] = exp( -eta_t * ( L[a] - min(L) ) ) for a in {0,1}
         p[a] = w[a] / sum_a' w[a']
         sample A_t from p
         measure mistake if A_t != X[t-1]
         update L[a] = L[a] + I{expert a is wrong}

    Return the array of cumulative losses (length T).
    """
    T = len(X)
    L = np.zeros(2)  # L_0(0)=0, L_0(1)=0
    cum_loss = np.zeros(T)
    
    alg_mistakes = 0
    for t in range(1, T+1):
        # define eta_t
        if schedule_type == 'fixed':
            eta_t = param
        elif schedule_type == 'anytime':
            eta_t = param * np.sqrt(np.log(2)/t)
        else:
            raise ValueError("schedule_type must be 'fixed' or 'anytime'")
        
        # shift by min(L) for numerical stability
        L_min = np.min(L)
        w = np.exp(-eta_t * (L - L_min))
        w_sum = np.sum(w)
        p = w / w_sum
        
        # sample action from p
        action = np.random.choice([0,1], p=p)
        
        # measure mistake
        if action != X[t-1]:
            alg_mistakes += 1
        
        # update experts' losses
        loss0 = 1 if (X[t-1] == 1) else 0
        loss1 = 1 if (X[t-1] == 0) else 0
        L[0] += loss0
        L[1] += loss1
        
        cum_loss[t-1] = alg_mistakes
    
    return cum_loss

def run_single_experiment(X, eta_fixed_list, c_anytime_list):
    """
    Run FTL + Hedge (with the given lists of parameters) on a single sequence X.
    Return a dict: alg_name -> array of length T (cumulative losses).
    """
    T = len(X)
    results = {}

    # FTL
    results["FTL"] = ftl_predict(X)
    
    # Hedge with fixed etas
    for eta in eta_fixed_list:
        name = f"Hedge_fixed({eta})"
        results[name] = hedge_predict(X, 'fixed', eta)
    
    # Hedge with anytime
    for c_val in c_anytime_list:
        name = f"Hedge_anytime({c_val})"
        results[name] = hedge_predict(X, 'anytime', c_val)
    
    return results

def run_experiments_for_p(p, T=2000, n_runs=10, 
                          eta_fixed_list=[0.5, 1.0],
                          c_anytime_list=[0.5, 1.0]):
    """
    For a given p, run n_runs i.i.d. Bernoulli(p) sequences of length T,
    apply FTL + the Hedge variants with parameters from the lists,
    compute average (and stdev) pseudo-regret as a function of t.

    The exercise text: 
    - T=2000
    - multiple mu (p) values
    - 10 runs
    - Hedge with the requested etas 
    """
    alg_names = ["FTL"] + \
                [f"Hedge_fixed({eta})" for eta in eta_fixed_list] + \
                [f"Hedge_anytime({c_val})" for c_val in c_anytime_list]
    
    # We'll store cumulative losses for each run: shape (n_runs, T)
    all_cum_loss = {name: [] for name in alg_names}
    all_best_loss = []
    
    for _ in range(n_runs):
        # generate data
        X = simulate_sequence(T, p)
        
        # best expert
        best_l = best_expert_loss(X)
        all_best_loss.append(best_l)
        
        # run each algorithm
        single_run_results = run_single_experiment(X, eta_fixed_list, c_anytime_list)
        for name in alg_names:
            all_cum_loss[name].append(single_run_results[name])
    
    # convert to arrays
    for name in alg_names:
        all_cum_loss[name] = np.array(all_cum_loss[name])  # shape (n_runs, T)
    all_best_loss = np.array(all_best_loss)               # shape (n_runs, T)
    
    # compute pseudo-regret per run
    # pseudo_regret[run, t] = all_cum_loss[name][run, t] - all_best_loss[run, t]
    avg_pseudo_regret = {}
    std_pseudo_regret = {}
    for name in alg_names:
        diff = all_cum_loss[name] - all_best_loss
        mean_diff = np.mean(diff, axis=0)
        std_diff  = np.std(diff, axis=0)
        avg_pseudo_regret[name] = mean_diff
        std_pseudo_regret[name] = std_diff
    
    return avg_pseudo_regret, std_pseudo_regret, alg_names

def main():
    T = 2000
    n_runs = 10

    # According to the instructions
    p_list = [
        0.5 - 1/4,   # 0.25
        0.5 - 1/8,   # 0.375
        0.5 - 1/16,  # 0.4375
    ]

    # For T=2000, K=2, so ln(K) = ln(2)
    #  -- "from class" = sqrt(2 ln(2) / T)
    #  -- "from ex. 5.7" = sqrt(8 ln(2) / T)

    eta_fixed_list = [
        np.sqrt(2.0 * np.log(2) / T),   # from class
        np.sqrt(8.0 * np.log(2) / T)    # from Exercise 5.7
    ]

    # For the anytime schedule:
    #  -- simple analysis => param=1 => eta_t = 1 * sqrt( ln(2)/ t )
    #  -- ex. 5.7        => param=2 => eta_t = 2 * sqrt( ln(2)/ t )

    c_anytime_list = [
        1.0,  # simple analysis
        2.0   # from Exercise 5.7
    ]


    for p in p_list:
        avg_pr, std_pr, alg_names = run_experiments_for_p(
            p=p,
            T=T,
            n_runs=n_runs,
            eta_fixed_list=eta_fixed_list,
            c_anytime_list=c_anytime_list
        )
        
        # Plot
        plt.figure(figsize=(9,6))
        t_range = np.arange(1, T+1)
        for name in alg_names:
            mean_vals = avg_pr[name]
            std_vals  = std_pr[name]
            plt.plot(t_range, mean_vals, label=name)
            plt.fill_between(t_range, mean_vals-std_vals, mean_vals+std_vals, alpha=0.2)
        
        plt.title(f"Exercise 5.8 - p={p}, T={T}, n_runs={n_runs}")
        plt.xlabel("t")
        plt.ylabel("Pseudo-Regret (mean +/- 1 std)")
        plt.grid(True)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
