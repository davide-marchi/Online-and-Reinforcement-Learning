import numpy as np
import matplotlib.pyplot as plt

def simulate_sequence(T, p):
    """
    Generate a length-T binary sequence i.i.d. Bernoulli(p).
    """
    return np.random.binomial(n=1, p=p, size=T)

def simulate_adversarial_sequence(T):
    """
    Generate a length-T adversarial (non-i.i.d.) sequence that alternates 0 and 1.
    This simple alternating sequence forces FTL to change frequently.
    """
    return np.array([t % 2 for t in range(T)])

def best_expert_loss(X):
    """
    For 2 experts (always predict 0 or always predict 1):
    best_expert_loss[t] = min(number of 1's in first t steps, number of 0's in first t steps).
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
    At time t, pick the expert with fewer mistakes so far.
    Returns the cumulative loss array (length T).
    """
    T = len(X)
    cum_loss = np.zeros(T)
    L0, L1 = 0, 0   # mistakes for expert0 and expert1
    alg_mistakes = 0
    for t in range(T):
        if L0 < L1:
            pred = 0
        elif L1 < L0:
            pred = 1
        else:
            pred = 0  # tie-breaker: choose 0
        if pred != X[t]:
            alg_mistakes += 1
        if X[t] == 1:
            L0 += 1
        else:
            L1 += 1
        cum_loss[t] = alg_mistakes
    return cum_loss

def hedge_predict(X, schedule_type, param):
    """
    Hedge algorithm for 2 experts (0 always predicts 0, 1 always predicts 1).
    For t=1,...,T:
      - If schedule_type == 'fixed':   eta_t = param
      - If schedule_type == 'anytime': eta_t = param * sqrt(ln(2)/t)
      - For numerical stability, shift losses by min(L).
    Returns the cumulative loss array (length T).
    """
    T = len(X)
    L = np.zeros(2)  # losses for the two experts
    cum_loss = np.zeros(T)
    alg_mistakes = 0
    for t in range(1, T+1):
        if schedule_type == 'fixed':
            eta_t = param
        elif schedule_type == 'anytime':
            eta_t = param * np.sqrt(np.log(2)/t)
        else:
            raise ValueError("schedule_type must be 'fixed' or 'anytime'")
        L_min = np.min(L)
        w = np.exp(-eta_t * (L - L_min))
        p = w / np.sum(w)
        action = np.random.choice([0, 1], p=p)
        if action != X[t-1]:
            alg_mistakes += 1
        loss0 = 1 if X[t-1] == 1 else 0
        loss1 = 1 if X[t-1] == 0 else 0
        L[0] += loss0
        L[1] += loss1
        cum_loss[t-1] = alg_mistakes
    return cum_loss

def run_single_experiment(X, eta_fixed_list, c_anytime_list):
    """
    Runs FTL and Hedge (with both fixed and anytime schedules) on sequence X.
    Returns a dictionary mapping algorithm names to their cumulative loss arrays.
    """
    results = {}
    results["FTL"] = ftl_predict(X)
    for eta in eta_fixed_list:
        name = f"Hedge_fixed({eta})"
        results[name] = hedge_predict(X, 'fixed', eta)
    for c in c_anytime_list:
        name = f"Hedge_anytime({c})"
        results[name] = hedge_predict(X, 'anytime', c)
    return results

def run_experiments_for_p(p, T=2000, n_runs=10, eta_fixed_list=None, c_anytime_list=None):
    """
    For a given p, run n_runs i.i.d. experiments (each a Bernoulli(p) sequence of length T),
    and compute the pseudo-regret (algorithm loss minus best expert loss) averaged over runs.
    Returns dictionaries for average and standard deviation of the pseudo-regret curves, and the list of algorithm names.
    """
    if eta_fixed_list is None:
        eta_fixed_list = []
    if c_anytime_list is None:
        c_anytime_list = []
    alg_names = (["FTL"] +
                 [f"Hedge_fixed({eta})" for eta in eta_fixed_list] +
                 [f"Hedge_anytime({c})" for c in c_anytime_list])
    all_cum_loss = {name: [] for name in alg_names}
    all_best_loss = []
    for _ in range(n_runs):
        X = simulate_sequence(T, p)
        best_l = best_expert_loss(X)
        all_best_loss.append(best_l)
        results = run_single_experiment(X, eta_fixed_list, c_anytime_list)
        for name in alg_names:
            all_cum_loss[name].append(results[name])
    for name in alg_names:
        all_cum_loss[name] = np.array(all_cum_loss[name])  # shape (n_runs, T)
    all_best_loss = np.array(all_best_loss)               # shape (n_runs, T)
    
    avg_pseudo_regret = {}
    std_pseudo_regret = {}
    for name in alg_names:
        diff = all_cum_loss[name] - all_best_loss
        avg_pseudo_regret[name] = np.mean(diff, axis=0)
        std_pseudo_regret[name] = np.std(diff, axis=0)
    return avg_pseudo_regret, std_pseudo_regret, alg_names

def run_experiments_adversarial_regret(T=2000, n_runs=10, eta_fixed_list=None, c_anytime_list=None):
    """
    Runs n_runs experiments on a fixed adversarial sequence (alternating 0,1,...),
    computes the (actual) regret for each algorithm (cumulative loss minus best expert loss on that sequence),
    and returns the average and standard deviation of the regret curves and the algorithm names.
    """
    if eta_fixed_list is None:
        eta_fixed_list = []
    if c_anytime_list is None:
        c_anytime_list = []
    alg_names = (["FTL"] +
                 [f"Hedge_fixed({eta})" for eta in eta_fixed_list] +
                 [f"Hedge_anytime({c})" for c in c_anytime_list])
    
    X_adv = simulate_adversarial_sequence(T)
    best_l = best_expert_loss(X_adv)
    
    all_regrets = {name: [] for name in alg_names}
    for _ in range(n_runs):
        results = run_single_experiment(X_adv, eta_fixed_list, c_anytime_list)
        for name in alg_names:
            # Regret = algorithm's cumulative loss - best expert's loss on the adversarial sequence
            regret = results[name] - best_l
            all_regrets[name].append(regret)
    for name in alg_names:
        all_regrets[name] = np.array(all_regrets[name])
    
    avg_regret = {}
    std_regret = {}
    for name in alg_names:
        avg_regret[name] = np.mean(all_regrets[name], axis=0)
        std_regret[name] = np.std(all_regrets[name], axis=0)
    return avg_regret, std_regret, alg_names

def main():
    T = 2000
    n_runs = 10
    
    # p values as specified in the exercise:
    p_list = [0.5 - 1/4, 0.5 - 1/8, 0.5 - 1/16]  # 0.25, 0.375, 0.4375
    
    # For fixed Hedge: use the values from class and from Exercise 5.7.
    # For T=2000, with K=2 so ln(2) is used:
    eta_fixed_list = [np.sqrt(2 * np.log(2) / T), np.sqrt(8 * np.log(2) / T)]
    # For anytime Hedge:
    c_anytime_list = [1.0, 2.0]
    
    ########################################
    # 1. i.i.d. Experiments: Plot Pseudo-Regret
    ########################################
    for p in p_list:
        avg_pr, std_pr, alg_names = run_experiments_for_p(
            p=p,
            T=T,
            n_runs=n_runs,
            eta_fixed_list=eta_fixed_list,
            c_anytime_list=c_anytime_list
        )
        plt.figure(figsize=(9,6))
        t_range = np.arange(1, T+1)
        for name in alg_names:
            plt.plot(t_range, avg_pr[name], label=name)
            plt.fill_between(t_range, avg_pr[name] - std_pr[name],
                             avg_pr[name] + std_pr[name], alpha=0.2)
        plt.title(f"i.i.d. Experiment - p={p}, T={T}, n_runs={n_runs}")
        plt.xlabel("t")
        plt.ylabel("Pseudo-Regret (mean ± 1 std)")
        plt.grid(True)
        plt.legend()
        plt.show()
        
        # Print final average pseudo-regret for each algorithm
        print(f"\nFinal average pseudo-regret at t={T} for p={p}:")
        for name in alg_names:
            print(f"  {name}: {avg_pr[name][-1]:.3f}")
        print("-" * 50)
    
    ########################################
    # 2. Adversarial Experiments: Plot Regret
    ########################################
    avg_regret, std_regret, alg_names_adv = run_experiments_adversarial_regret(
        T=T,
        n_runs=n_runs,
        eta_fixed_list=eta_fixed_list,
        c_anytime_list=c_anytime_list
    )
    plt.figure(figsize=(9,6))
    t_range = np.arange(1, T+1)
    for name in alg_names_adv:
        plt.plot(t_range, avg_regret[name], label=name)
        plt.fill_between(t_range,
                         avg_regret[name] - std_regret[name],
                         avg_regret[name] + std_regret[name],
                         alpha=0.2)
    plt.title(f"Adversarial Experiment - Regret, T={T}, n_runs={n_runs}")
    plt.xlabel("t")
    plt.ylabel("Regret (mean ± 1 std)")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Print final average regret for each algorithm in the adversarial case
    print(f"\nFinal average regret at t={T} (adversarial):")
    for name in alg_names_adv:
        print(f"  {name}: {avg_regret[name][-1]:.3f}")

if __name__ == "__main__":
    main()
