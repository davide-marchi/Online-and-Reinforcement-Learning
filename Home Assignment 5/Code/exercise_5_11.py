import numpy as np
import matplotlib
matplotlib.use('Agg')  # Ensures we can save figures without displaying them
import matplotlib.pyplot as plt

###############################################################################
# 1) HELPER FUNCTIONS
###############################################################################

def bernoulli_sample(p):
    """Return 1 with probability p, else 0."""
    return 1 if (np.random.rand() < p) else 0

def run_ucb(T, means, version='modified', nonstationary=False):
    """
    Run the UCB1 algorithm (using the 'modified' bonus term) on a bandit problem.
    If nonstationary=True, then for arm 1 we force the reward probability to 0
    once t >= T/2 (this 'breaks' UCB in the adversarial setting).
    """
    n_arms = len(means)
    best_arm = np.argmax(means)
    # gap[a] = (best mean) - (mean of arm a), used to compute pseudo-regret
    gap = [means[best_arm] - means[a] for a in range(n_arms)]
    
    counts = np.zeros(n_arms, dtype=int)   # number of pulls for each arm
    values = np.zeros(n_arms)             # running average of rewards
    regrets = np.zeros(T)
    cumulative_regret = 0.0
    
    # Pull each arm once (to initialize)
    for a in range(n_arms):
        if nonstationary and a == 1 and a >= T/2:
            reward = bernoulli_sample(0.0)
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
            # "Modified" bonus: sqrt( (log t) / n_a )
            bonus = np.sqrt(np.log(current_time) / counts[a])
            ucb_values[a] = avg_reward + bonus
        
        chosen_arm = np.argmax(ucb_values)
        
        if nonstationary and chosen_arm == 1 and t >= T/2:
            reward = bernoulli_sample(0.0)
        else:
            reward = bernoulli_sample(means[chosen_arm])
        
        counts[chosen_arm] += 1
        # Update running average
        values[chosen_arm] += (reward - values[chosen_arm]) / counts[chosen_arm]
        
        cumulative_regret += gap[chosen_arm]
        regrets[t] = cumulative_regret
    
    return regrets

def run_exp3(T, means, nonstationary=False):
    """
    Run the EXP3 algorithm with time-varying learning rate
      eta_t = sqrt( (ln K) / (t * K) ).
    For rewards in [0,1], define loss = 1 - reward.
    If nonstationary=True, then for arm 1 we force reward probability = 0
    once t >= T/2.
    Returns the pseudo-regret at each time.
    """
    n_arms = len(means)
    best_mean = np.max(means)
    regrets = np.zeros(T)
    cumulative_regret = 0.0
    
    # L[a] = cumulative loss for arm a
    L = np.zeros(n_arms)
    
    for t in range(T):
        # time-varying learning rate
        eta_t = np.sqrt(np.log(n_arms) / ((t+1) * n_arms))
        
        # Subtract-min trick for numerical stability
        L_min = np.min(L)
        w = np.exp(-eta_t * (L - L_min))
        p = w / np.sum(w)
        
        chosen_arm = np.random.choice(n_arms, p=p)
        
        # Nonstationary "break" scenario:
        if nonstationary and chosen_arm == 1 and t >= T/2:
            reward = bernoulli_sample(0.0)
        else:
            reward = bernoulli_sample(means[chosen_arm])
        
        # Update pseudo-regret
        cumulative_regret += (best_mean - reward)
        regrets[t] = cumulative_regret
        
        # Update losses using importance weighting
        est_loss = (1 - reward) / max(p[chosen_arm], 1e-12)
        L[chosen_arm] += est_loss
    
    return regrets

###############################################################################
# 2) FULL IID EXPERIMENTS FOR K = 2,4,8,16 AND THREE GAP VALUES
###############################################################################

def run_experiments_iid(T=10_000, n_runs=20):
    """
    Run three different i.i.d. experiments (with different gap values Delta)
    for each K in {2,4,8,16}, comparing:
       - UCB1 (modified/improved parametrization)
       - EXP3
    We assume one best arm with mean mu^* = 0.5, suboptimal arms with mean = 0.5 - Delta.
    """
    K_values = [2, 4, 8, 16]
    deltas = [1/4, 1/8, 1/16]   # i.e. 0.25, 0.125, 0.0625
    
    # Create one figure per Delta, each figure containing 4 subplots (one for each K).
    for Delta in deltas:
        fig, axes = plt.subplots(1, len(K_values), figsize=(20,5), sharey=True)
        fig.suptitle(f"IID Bandits: Delta = {Delta}, (mu* = 0.5, suboptimal = 0.5 - {Delta})", fontsize=14)
        
        for i, K in enumerate(K_values):
            mu_star = 0.5
            mu_sub = mu_star - Delta
            # Build the means array: 1 best arm, K-1 suboptimal arms
            means = [mu_star] + [mu_sub]*(K-1)
            
            # Collect regrets for the algorithms:
            all_reg_ucb = np.zeros((n_runs, T))
            all_reg_exp3 = np.zeros((n_runs, T))
            
            for r in range(n_runs):
                # UCB1 with improved parametrization
                reg_ucb = run_ucb(T, means, version='modified', nonstationary=False)
                all_reg_ucb[r, :] = reg_ucb
                
                # EXP3
                reg_exp3 = run_exp3(T, means, nonstationary=False)
                all_reg_exp3[r, :] = reg_exp3
            
            # Compute means and standard deviations
            mean_ucb = np.mean(all_reg_ucb, axis=0)
            std_ucb  = np.std(all_reg_ucb, axis=0)
            
            mean_exp3 = np.mean(all_reg_exp3, axis=0)
            std_exp3  = np.std(all_reg_exp3, axis=0)
            
            ax = axes[i]
            t_axis = np.arange(1, T+1)
            
            ax.plot(t_axis, mean_ucb, label="UCB1 (modified)", color='green')
            ax.fill_between(t_axis,
                            mean_ucb - std_ucb,
                            mean_ucb + std_ucb,
                            alpha=0.2, color='green')
            
            ax.plot(t_axis, mean_exp3, label="EXP3", color='red')
            ax.fill_between(t_axis,
                            mean_exp3 - std_exp3,
                            mean_exp3 + std_exp3,
                            alpha=0.2, color='red')
            
            ax.set_title(f"K = {K}")
            ax.set_xlabel("t")
            if i == 0:
                ax.set_ylabel("Cumulative Pseudo-Regret")
            ax.legend()
        
        plt.tight_layout()
        delta_str = str(Delta).replace('.', '_')
        plt.savefig(f"Home Assignment 5/Code/iid_experiment_delta_{delta_str}.png")
        plt.close()

###############################################################################
# 3) BREAK (ADVERSARIAL) EXPERIMENT
###############################################################################

def run_experiment_break(T=10_000, n_runs=20):
    """
    Adversarial design to break UCB1:
      - 2 arms: arm 0 has mean 0.5 (constant),
                arm 1 has mean 0.9 for t < T/2 and 0.0 for t >= T/2.
    Compare UCB1 (modified) vs EXP3.
    """
    n_runs = n_runs
    all_regrets_ucb = np.zeros((n_runs, T))
    all_regrets_exp3 = np.zeros((n_runs, T))
    
    means = [0.5, 0.9]
    
    for r in range(n_runs):
        # UCB1 with improved parametrization
        reg_ucb = run_ucb(T, means, version='modified', nonstationary=True)
        all_regrets_ucb[r, :] = reg_ucb
        
        # EXP3
        reg_exp3 = run_exp3(T, means, nonstationary=True)
        all_regrets_exp3[r, :] = reg_exp3
    
    # Plotting average regrets +/- std
    mean_ucb = np.mean(all_regrets_ucb, axis=0)
    std_ucb  = np.std(all_regrets_ucb, axis=0)
    mean_exp3 = np.mean(all_regrets_exp3, axis=0)
    std_exp3  = np.std(all_regrets_exp3, axis=0)
    
    t_vals = np.arange(1, T+1)
    plt.figure(figsize=(7,5))
    plt.plot(t_vals, mean_ucb, label="UCB1 (modified)", color='green')
    plt.fill_between(t_vals, mean_ucb - std_ucb, mean_ucb + std_ucb,
                     alpha=0.2, color='green')
    
    plt.plot(t_vals, mean_exp3, label="EXP3", color='red')
    plt.fill_between(t_vals, mean_exp3 - std_exp3, mean_exp3 + std_exp3,
                     alpha=0.2, color='red')
    
    plt.title("Breaking UCB1 (Nonstationary/Adversarial Setting)")
    plt.xlabel("t")
    plt.ylabel("Cumulative Pseudo-Regret")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Home Assignment 5/Code/break_experiment.png")
    plt.close()

###############################################################################
# 4) MAIN: RUN EVERYTHING
###############################################################################
if __name__ == "__main__":
    # Set T=10,000 for demonstration; adjust T or n_runs as needed.
    T = 10_000
    n_runs = 20
    
    # 1) IID experiments for K=2,4,8,16 with deltas in {1/4, 1/8, 1/16}
    run_experiments_iid(T=T, n_runs=n_runs)
    
    # 2) Break experiment (adversarial)
    run_experiment_break(T=T, n_runs=n_runs)
