import numpy as np
import matplotlib
matplotlib.use('Agg')  # (optional) ensures we can save figures without showing
import matplotlib.pyplot as plt

###############################################################################
# 1) HELPER FUNCTIONS
###############################################################################

def bernoulli_sample(p):
    """Return 1 with probability p, else 0."""
    return 1 if (np.random.rand() < p) else 0

def run_ucb(T, means, version='original', nonstationary=False):
    """
    Run the UCB1 algorithm (either 'original' or 'modified') on a bandit problem.
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
            if version == 'original':
                # Typical UCB1: bonus ~ sqrt( (2 log t) / n_a )
                # or you can keep your 1.5 factor, etc.
                bonus = np.sqrt(1.5 * np.log(current_time) / counts[a])
            else:
                # "Improved" or “modified” bonus, e.g. sqrt( (log t) / n_a )
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
        
        # Update losses
        est_loss = (1 - reward) / max(p[chosen_arm], 1e-12)
        L[chosen_arm] += est_loss
    
    return regrets

###############################################################################
# 2) FULL IID EXPERIMENTS FOR K = 2,4,8,16 AND THREE GAP VALUES
###############################################################################

def run_experiments_iid(T=10_000, n_runs=20):
    """
    Run *three* different i.i.d. experiments (with different gaps Delta)
    for each K in {2,4,8,16}, comparing:
       - UCB1 (original)
       - UCB1 (modified/improved)
       - EXP3
    We assume one best arm with mean mu^* = 0.5, suboptimal arms = mu^* - Delta.
    """
    K_values = [2, 4, 8, 16]
    deltas = [1/4, 1/8, 1/16]   # i.e. 0.25, 0.125, 0.0625
    
    # We'll create one figure per Delta, each figure containing 4 subplots (one for each K).
    for Delta_idx, Delta in enumerate(deltas):
        fig, axes = plt.subplots(1, len(K_values), figsize=(20,5), sharey=True)
        fig.suptitle(f"IID Bandits: Delta = {Delta},  (mu* = 0.5,  suboptimal = 0.5 - {Delta})", fontsize=14)
        
        for i, K in enumerate(K_values):
            mu_star = 0.5
            mu_sub = mu_star - Delta
            # Build the means array: 1 best arm, K-1 suboptimal arms
            means = [mu_star] + [mu_sub]*(K-1)
            
            # We'll collect regrets for each algorithm:
            all_reg_ucb_orig = np.zeros((n_runs, T))
            all_reg_ucb_mod  = np.zeros((n_runs, T))
            all_reg_exp3     = np.zeros((n_runs, T))
            
            for r in range(n_runs):
                # UCB1 (original)
                reg_ucb_o = run_ucb(T, means, version='original', nonstationary=False)
                all_reg_ucb_orig[r,:] = reg_ucb_o
                
                # UCB1 (modified/improved)
                reg_ucb_m = run_ucb(T, means, version='modified', nonstationary=False)
                all_reg_ucb_mod[r,:] = reg_ucb_m
                
                # EXP3
                reg_exp3 = run_exp3(T, means, nonstationary=False)
                all_reg_exp3[r,:] = reg_exp3
            
            # Compute means + std
            mean_ucb_o = np.mean(all_reg_ucb_orig, axis=0)
            std_ucb_o  = np.std(all_reg_ucb_orig, axis=0)
            
            mean_ucb_m = np.mean(all_reg_ucb_mod, axis=0)
            std_ucb_m  = np.std(all_reg_ucb_mod, axis=0)
            
            mean_exp3  = np.mean(all_reg_exp3, axis=0)
            std_exp3   = np.std(all_reg_exp3, axis=0)
            
            ax = axes[i]
            t_axis = np.arange(1, T+1)
            
            ax.plot(t_axis, mean_ucb_o, label="UCB1 (original)", color='blue')
            ax.fill_between(t_axis,
                            mean_ucb_o - std_ucb_o,
                            mean_ucb_o + std_ucb_o,
                            alpha=0.2, color='blue')
            
            ax.plot(t_axis, mean_ucb_m, label="UCB1 (modified)", color='green')
            ax.fill_between(t_axis,
                            mean_ucb_m - std_ucb_m,
                            mean_ucb_m + std_ucb_m,
                            alpha=0.2, color='green')
            
            ax.plot(t_axis, mean_exp3,  label="EXP3", color='red')
            ax.fill_between(t_axis,
                            mean_exp3 - std_exp3,
                            mean_exp3 + std_exp3,
                            alpha=0.2, color='red')
            
            ax.set_title(f"K = {K}")
            ax.set_xlabel("t")
            if i == 0:
                ax.set_ylabel("Cumulative Pseudo-Regret")
            ax.legend()
        
        # Save figure with a filename that includes Delta
        plt.tight_layout()
        # Replace '.' with '_' in Delta for a safer filename
        delta_str = str(Delta).replace('.', '_')
        plt.savefig(f"iid_experiment_delta_{delta_str}.png")
        plt.close()  # Close the figure so it doesn't display

###############################################################################
# 3) BREAK (ADVERSARIAL) EXPERIMENT
###############################################################################

def run_experiment_break(T=10_000, n_runs=20):
    """
    Adversarial design to break UCB1:
      - 2 arms: arm 0 has mean 0.5 (constant),
                arm 1 has mean 0.9 for t < T/2 and 0.0 for t >= T/2.
    Compare UCB1 vs EXP3. Show average regrets +/- std over n_runs.
    """
    # We keep exactly 2 arms, with initial means [0.5, 0.9].
    # The 'nonstationary=True' flag in run_ucb / run_exp3
    # forces the second arm to become 0.0 after T/2.
    
    all_regrets_ucb = np.zeros((n_runs, T))
    all_regrets_exp3 = np.zeros((n_runs, T))
    
    means = [0.5, 0.9]
    
    for r in range(n_runs):
        # UCB1
        reg_ucb = run_ucb(T, means, version='original', nonstationary=True)
        all_regrets_ucb[r,:] = reg_ucb
        
        # EXP3
        reg_e3 = run_exp3(T, means, nonstationary=True)
        all_regrets_exp3[r,:] = reg_e3
    
    # Plot
    mean_ucb = np.mean(all_regrets_ucb, axis=0)
    std_ucb  = np.std(all_regrets_ucb, axis=0)
    mean_e3  = np.mean(all_regrets_exp3, axis=0)
    std_e3   = np.std(all_regrets_exp3, axis=0)
    
    t_vals = np.arange(1, T+1)
    plt.figure(figsize=(7,5))
    plt.plot(t_vals, mean_ucb, label="UCB1", color='blue')
    plt.fill_between(t_vals, mean_ucb - std_ucb, mean_ucb + std_ucb,
                     alpha=0.2, color='blue')
    
    plt.plot(t_vals, mean_e3, label="EXP3", color='red')
    plt.fill_between(t_vals, mean_e3 - std_e3, mean_e3 + std_e3,
                     alpha=0.2, color='red')
    
    plt.title("Breaking UCB1 in a Nonstationary/Adversarial Setting")
    plt.xlabel("t")
    plt.ylabel("Cumulative Pseudo-Regret")
    plt.legend()
    plt.tight_layout()
    plt.savefig("break_experiment.png")
    plt.close()  # Close the figure

###############################################################################
# 4) MAIN: RUN EVERYTHING
###############################################################################
if __name__ == "__main__":
    # You can set T=100_000 for the full experiment, but it may be slower.
    # For demonstration, T=10_000 is typically fast enough. Adjust as needed.
    T = 10_000
    n_runs = 20
    
    # 1) IID experiments for K=2,4,8,16 and deltas = {1/4,1/8,1/16}
    run_experiments_iid(T=T, n_runs=n_runs)
    
    # 2) Break experiment (adversarial)
    run_experiment_break(T=T, n_runs=n_runs)
