import numpy as np
import matplotlib
matplotlib.use('Agg')  # Ensures we can save figures without displaying them
import matplotlib.pyplot as plt

###############################################################################
# 0) ADVERSARIAL REWARD FUNCTION (for worst-case break experiment)
###############################################################################
def adversarial_reward(t, arm, T, epsilon=0.01):
    """
    Returns deterministic (or nearly deterministic) rewards for the break experiment.
    For t < T/2: arm 0 yields 0 and arm 1 yields 1.
    For t >= T/2: arm 0 yields 1 and arm 1 yields 1 with probability epsilon, else 0.
    """
    if t < T/2:
        return 0 if arm == 0 else 1
    else:
        return 1 if arm == 0 else (1 if np.random.rand() < epsilon else 0)

###############################################################################
# 1) HELPER FUNCTIONS
###############################################################################
def bernoulli_sample(p):
    """Return 1 with probability p, else 0."""
    return 1 if (np.random.rand() < p) else 0

def run_ucb(T, means, version='modified', nonstationary=False, worst_case=False, epsilon=0.01):
    """
    Run UCB1 (using the 'modified' bonus: sqrt(log t / n_a)) on a bandit problem.
    If nonstationary is True, then the rewards change after t >= T/2.
    If worst_case is True, we use a worst-case adversarial reward function (see adversarial_reward).
    """
    n_arms = len(means)
    best_arm = np.argmax(means)
    gap = [means[best_arm] - means[a] for a in range(n_arms)]
    
    counts = np.zeros(n_arms, dtype=int)   # pulls per arm
    values = np.zeros(n_arms)             # running average rewards
    regrets = np.zeros(T)
    cumulative_regret = 0.0
    
    # Initialization: pull each arm once.
    for a in range(n_arms):
        if nonstationary and worst_case:
            # For initialization, t is taken as a (which is < T/2 if n_arms is small)
            reward = adversarial_reward(t=a, arm=a, T=T, epsilon=epsilon)
        elif nonstationary:
            # Original nonstationary: for arm 1, if a >= T/2 then force reward 0.
            if a == 1 and a >= T/2:
                reward = bernoulli_sample(0.0)
            else:
                reward = bernoulli_sample(means[a])
        else:
            reward = bernoulli_sample(means[a])
        counts[a] = 1
        values[a] = reward
        cumulative_regret += gap[a]
        regrets[a] = cumulative_regret
    
    # Main loop
    for t in range(n_arms, T):
        current_time = t + 1  # because t starts at 0
        ucb_values = np.zeros(n_arms)
        
        for a in range(n_arms):
            avg_reward = values[a]
            bonus = np.sqrt(np.log(current_time) / counts[a])
            ucb_values[a] = avg_reward + bonus
        
        chosen_arm = np.argmax(ucb_values)
        
        if nonstationary and worst_case:
            reward = adversarial_reward(t, chosen_arm, T, epsilon)
        elif nonstationary:
            if chosen_arm == 1 and t >= T/2:
                reward = bernoulli_sample(0.0)
            else:
                reward = bernoulli_sample(means[chosen_arm])
        else:
            reward = bernoulli_sample(means[chosen_arm])
        
        counts[chosen_arm] += 1
        values[chosen_arm] += (reward - values[chosen_arm]) / counts[chosen_arm]
        cumulative_regret += gap[chosen_arm]
        regrets[t] = cumulative_regret
    
    return regrets

def run_exp3(T, means, nonstationary=False, worst_case=False, epsilon=0.01):
    """
    Run the EXP3 algorithm with time-varying learning rate eta_t = sqrt(log(K)/(t*K)).
    For rewards in [0,1], define loss = 1 - reward.
    If nonstationary and worst_case are True, use adversarial_reward.
    """
    n_arms = len(means)
    best_mean = np.max(means)
    regrets = np.zeros(T)
    cumulative_regret = 0.0
    L = np.zeros(n_arms)  # cumulative losses
    
    for t in range(T):
        eta_t = np.sqrt(np.log(n_arms) / ((t+1) * n_arms))
        L_min = np.min(L)
        w = np.exp(-eta_t * (L - L_min))
        p = w / np.sum(w)
        chosen_arm = np.random.choice(n_arms, p=p)
        
        if nonstationary and worst_case:
            reward = adversarial_reward(t, chosen_arm, T, epsilon)
        elif nonstationary:
            if chosen_arm == 1 and t >= T/2:
                reward = bernoulli_sample(0.0)
            else:
                reward = bernoulli_sample(means[chosen_arm])
        else:
            reward = bernoulli_sample(means[chosen_arm])
        
        cumulative_regret += (best_mean - reward)
        regrets[t] = cumulative_regret
        est_loss = (1 - reward) / max(p[chosen_arm], 1e-12)
        L[chosen_arm] += est_loss
    
    return regrets

###############################################################################
# 2) IID EXPERIMENTS (unchanged)
###############################################################################
def run_experiments_iid(T=10_000, n_runs=20):
    """
    Run IID experiments (with different gap values) for each K in {2,4,8,16} comparing:
       - UCB1 (modified)
       - EXP3
    We assume one best arm with mean 0.5 and suboptimal arms with mean 0.5 - Delta.
    """
    K_values = [2, 4, 8, 16]
    deltas = [1/4, 1/8, 1/16]
    
    for Delta in deltas:
        fig, axes = plt.subplots(1, len(K_values), figsize=(20,5), sharey=True)
        fig.suptitle(f"IID Bandits: Delta = {Delta} (mu* = 0.5, suboptimal = 0.5 - {Delta})", fontsize=14)
        
        for i, K in enumerate(K_values):
            mu_star = 0.5
            mu_sub = mu_star - Delta
            means = [mu_star] + [mu_sub]*(K-1)
            
            all_reg_ucb = np.zeros((n_runs, T))
            all_reg_exp3 = np.zeros((n_runs, T))
            
            for r in range(n_runs):
                reg_ucb = run_ucb(T, means, version='modified', nonstationary=False)
                all_reg_ucb[r, :] = reg_ucb
                reg_exp3 = run_exp3(T, means, nonstationary=False)
                all_reg_exp3[r, :] = reg_exp3
            
            mean_ucb = np.mean(all_reg_ucb, axis=0)
            std_ucb  = np.std(all_reg_ucb, axis=0)
            mean_exp3 = np.mean(all_reg_exp3, axis=0)
            std_exp3  = np.std(all_reg_exp3, axis=0)
            
            ax = axes[i]
            t_axis = np.arange(1, T+1)
            ax.plot(t_axis, mean_ucb, label="UCB1 (modified)", color='green')
            ax.fill_between(t_axis, mean_ucb - std_ucb, mean_ucb + std_ucb, alpha=0.2, color='green')
            ax.plot(t_axis, mean_exp3, label="EXP3", color='red')
            ax.fill_between(t_axis, mean_exp3 - std_exp3, mean_exp3 + std_exp3, alpha=0.2, color='red')
            ax.set_title(f"K = {K}")
            ax.set_xlabel("t")
            if i == 0:
                ax.set_ylabel("Cumulative Pseudo-Regret")
            ax.legend()
        
        plt.tight_layout()
        delta_str = str(Delta).replace('.', '_')
        plt.savefig(f"iid_experiment_delta_{delta_str}.png")
        plt.close()

###############################################################################
# 3) WORST-CASE BREAK (ADVERSARIAL) EXPERIMENT
###############################################################################
def run_experiment_break(T=10_000, n_runs=20, epsilon=0.01):
    """
    Create a worst-case adversarial environment to break UCB:
      - 2 arms.
      For t < T/2:
          • Arm 0: reward = 0
          • Arm 1: reward = 1
      For t >= T/2:
          • Arm 0: reward = 1  (the new best arm)
          • Arm 1: reward = 1 with probability epsilon, else 0.
    This forces UCB (which has been front-loaded on arm 1) to suffer linear regret.
    Compare UCB1 (modified) vs EXP3.
    """
    n_runs = n_runs
    all_regrets_ucb = np.zeros((n_runs, T))
    all_regrets_exp3 = np.zeros((n_runs, T))
    
    # For the break experiment, we set the "nominal" means to be [0.5, 0.9] for reference,
    # but the rewards are overridden by adversarial_reward.
    means = [0.5, 0.9]
    
    for r in range(n_runs):
        # Use worst_case=True so that adversarial_reward is used.
        reg_ucb = run_ucb(T, means, version='modified', nonstationary=True, worst_case=True, epsilon=epsilon)
        all_regrets_ucb[r, :] = reg_ucb
        reg_exp3 = run_exp3(T, means, nonstationary=True, worst_case=True, epsilon=epsilon)
        all_regrets_exp3[r, :] = reg_exp3
    
    mean_ucb = np.mean(all_regrets_ucb, axis=0)
    std_ucb  = np.std(all_regrets_ucb, axis=0)
    mean_exp3 = np.mean(all_regrets_exp3, axis=0)
    std_exp3  = np.std(all_regrets_exp3, axis=0)
    
    t_vals = np.arange(1, T+1)
    plt.figure(figsize=(7,5))
    plt.plot(t_vals, mean_ucb, label="UCB1 (modified)", color='green')
    plt.fill_between(t_vals, mean_ucb - std_ucb, mean_ucb + std_ucb, alpha=0.2, color='green')
    plt.plot(t_vals, mean_exp3, label="EXP3", color='red')
    plt.fill_between(t_vals, mean_exp3 - std_exp3, mean_exp3 + std_exp3, alpha=0.2, color='red')
    
    plt.title("Worst-Case Breaking UCB (Adversarial Setting)")
    plt.xlabel("t")
    plt.ylabel("Cumulative Pseudo-Regret")
    plt.legend()
    plt.tight_layout()
    plt.savefig("break_experiment_worst_case.png")
    plt.close()

###############################################################################
# 4) MAIN: RUN EVERYTHING
###############################################################################
if __name__ == "__main__":
    T = 10_000
    n_runs = 20
    run_experiments_iid(T=T, n_runs=n_runs)
    run_experiment_break(T=T, n_runs=n_runs, epsilon=0.01)
