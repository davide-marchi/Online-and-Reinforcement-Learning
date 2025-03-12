import numpy as np
import matplotlib
matplotlib.use('Agg')  # so we can save figures without displaying them
import matplotlib.pyplot as plt

###############################################################################
# 0) ADVERSARIAL REWARD FUNCTION
###############################################################################
def adversarial_reward(t, arm, T, epsilon=0.01):
    """
    Worst-case reward schedule:
      - For t < T/2:
          Arm 0: reward = 0
          Arm 1: reward = 1
      - For t >= T/2:
          Arm 0: reward = 1
          Arm 1: reward = 1 with probability epsilon, else 0
    """
    if t < T/2:
        # first half: only arm 1 is good
        return 1 if arm == 1 else 0
    else:
        # second half: only arm 0 is good (arm 1 is almost always bad)
        if arm == 0:
            return 1
        else:
            return 1 if np.random.rand() < epsilon else 0

###############################################################################
# 1) HELPER FUNCTIONS
###############################################################################
def bernoulli_sample(p):
    """Returns 1 with probability p, else 0."""
    return 1 if (np.random.rand() < p) else 0

def run_ucb(T, means, version='modified', nonstationary=False,
            worst_case=False, epsilon=0.01):
    """
    Runs the 'modified' UCB1 algorithm:
      - bonus = sqrt((log t) / n_arm)
    If worst_case=True and nonstationary=True, uses 'adversarial_reward' instead
    of simple Bernoulli with means[].
    """
    n_arms = len(means)
    best_arm = np.argmax(means)
    # gap[a] = difference between best mean and the mean of arm a
    gap = [means[best_arm] - means[a] for a in range(n_arms)]
    
    counts = np.zeros(n_arms, dtype=int)  # how many times each arm was played
    values = np.zeros(n_arms)            # empirical average of each arm
    regrets = np.zeros(T)
    cumulative_regret = 0.0
    
    # Initialize by pulling each arm once
    for a in range(n_arms):
        if nonstationary and worst_case:
            reward = adversarial_reward(a, a, T, epsilon)
        elif nonstationary:
            # If it's the "simple" break scenario: arm 1 => reward=0 if a>=T/2
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
        current_time = t + 1
        ucb_values = np.zeros(n_arms)
        
        # Compute UCB indices
        for a in range(n_arms):
            avg_reward = values[a]
            bonus = np.sqrt(np.log(current_time) / counts[a])
            ucb_values[a] = avg_reward + bonus
        
        # Choose the arm with the highest UCB
        chosen_arm = np.argmax(ucb_values)
        
        # Get the reward depending on the scenario
        if nonstationary and worst_case:
            reward = adversarial_reward(t, chosen_arm, T, epsilon)
        elif nonstationary:
            if chosen_arm == 1 and t >= T/2:
                reward = bernoulli_sample(0.0)
            else:
                reward = bernoulli_sample(means[chosen_arm])
        else:
            reward = bernoulli_sample(means[chosen_arm])
        
        # Update counts and empirical average
        counts[chosen_arm] += 1
        values[chosen_arm] += (reward - values[chosen_arm]) / counts[chosen_arm]
        
        # Update cumulative regret
        cumulative_regret += gap[chosen_arm]
        regrets[t] = cumulative_regret
    
    return regrets

def run_exp3(T, means, nonstationary=False, worst_case=False, epsilon=0.01):
    """
    Runs EXP3 with time-varying learning rate:
      eta_t = sqrt(log(K) / (t*K))
    If worst_case=True (and nonstationary=True), uses adversarial_reward.
    """
    n_arms = len(means)
    best_mean = np.max(means)  # might not match actual adv. scenario, can cause negative regrets
    regrets = np.zeros(T)
    cumulative_regret = 0.0
    
    # L[a] is cumulative loss for each arm
    L = np.zeros(n_arms)
    
    for t in range(T):
        # compute learning rate
        eta_t = np.sqrt(np.log(n_arms) / ((t+1) * n_arms))
        
        # subtract-min trick for stability
        L_min = np.min(L)
        w = np.exp(-eta_t * (L - L_min))
        p = w / np.sum(w)
        
        # pick an arm according to p
        chosen_arm = np.random.choice(n_arms, p=p)
        
        # get the reward from environment
        if nonstationary and worst_case:
            reward = adversarial_reward(t, chosen_arm, T, epsilon)
        elif nonstationary:
            if chosen_arm == 1 and t >= T/2:
                reward = bernoulli_sample(0.0)
            else:
                reward = bernoulli_sample(means[chosen_arm])
        else:
            reward = bernoulli_sample(means[chosen_arm])
        
        # update pseudo-regret
        cumulative_regret += (best_mean - reward)
        regrets[t] = cumulative_regret
        
        # update losses for chosen_arm (1 - reward) with importance weighting
        loss = (1 - reward) / max(p[chosen_arm], 1e-12)
        L[chosen_arm] += loss
    
    return regrets

###############################################################################
# 2) IID EXPERIMENTS
###############################################################################
def run_experiments_iid(T=10_000, n_runs=20):
    """
    Runs i.i.d. experiments for K in {2,4,8,16} and Delta in {1/4,1/8,1/16}.
    Compares:
      - UCB1 (modified)
      - EXP3
    We assume one best arm with mean 0.5, suboptimal arms with mean = 0.5 - Delta.
    """
    K_values = [2, 4, 8, 16]
    deltas = [1/4, 1/8, 1/16]
    
    for Delta in deltas:
        fig, axes = plt.subplots(1, len(K_values), figsize=(20,5), sharey=True)
        fig.suptitle(f"IID Bandits: Delta = {Delta} (mu* = 0.5, sub = 0.5 - {Delta})")
        
        for i, K in enumerate(K_values):
            mu_star = 0.5
            mu_sub = mu_star - Delta
            means = [mu_star] + [mu_sub]*(K-1)
            
            all_reg_ucb = np.zeros((n_runs, T))
            all_reg_exp3 = np.zeros((n_runs, T))
            
            for r in range(n_runs):
                reg_ucb = run_ucb(T, means, version='modified', nonstationary=False)
                all_reg_ucb[r,:] = reg_ucb
                
                reg_e3 = run_exp3(T, means, nonstationary=False)
                all_reg_exp3[r,:] = reg_e3
            
            mean_ucb = np.mean(all_reg_ucb, axis=0)
            std_ucb  = np.std(all_reg_ucb, axis=0)
            mean_exp3 = np.mean(all_reg_exp3, axis=0)
            std_exp3  = np.std(all_reg_exp3, axis=0)
            
            ax = axes[i]
            t_axis = np.arange(1, T+1)
            ax.plot(t_axis, mean_ucb, label="UCB1 (mod)", color='green')
            ax.fill_between(t_axis, mean_ucb - std_ucb, mean_ucb + std_ucb,
                            alpha=0.2, color='green')
            ax.plot(t_axis, mean_exp3, label="EXP3", color='red')
            ax.fill_between(t_axis, mean_exp3 - std_exp3, mean_exp3 + std_exp3,
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
# 3) WORST-CASE BREAK EXPERIMENT
###############################################################################
def run_experiment_break(T=10_000, n_runs=20, epsilon=0.01):
    """
    This environment is designed to break UCB:
      - 2 arms total.
      - For t < T/2, arm 1 always returns 1, arm 0 returns 0.
      - For t >= T/2, arm 0 always returns 1, arm 1 returns 1 w/ prob epsilon.
    UCB invests heavily in arm 1 in the first half and doesn't adapt well
    in the second half. Meanwhile, EXP3 reweights faster.
    """
    all_regrets_ucb = np.zeros((n_runs, T))
    all_regrets_exp3 = np.zeros((n_runs, T))
    
    # The "means" array is just for reference. The real reward function is adversarial_reward.
    means = [0.5, 0.9]
    
    for r in range(n_runs):
        reg_ucb = run_ucb(T, means, version='modified',
                          nonstationary=True, worst_case=True, epsilon=epsilon)
        all_regrets_ucb[r,:] = reg_ucb
        
        reg_e3 = run_exp3(T, means, nonstationary=True, worst_case=True, epsilon=epsilon)
        all_regrets_exp3[r,:] = reg_e3
    
    # Averages
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
    
    plt.title("Worst-Case Breaking UCB (Adversarial Setting)")
    plt.xlabel("t")
    plt.ylabel("Cumulative Pseudo-Regret")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Home Assignment 5/Code/break_experiment_worst_case.png")
    plt.close()

###############################################################################
# 4) MAIN
###############################################################################
if __name__ == "__main__":
    T = 10_000
    n_runs = 20
    
    # Run the IID experiments first
    run_experiments_iid(T=T, n_runs=n_runs)
    
    # Then run the worst-case break experiment
    run_experiment_break(T=T, n_runs=n_runs, epsilon=0.01)
