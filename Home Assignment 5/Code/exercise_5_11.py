import numpy as np
import matplotlib.pyplot as plt

def bernoulli_sample(p):
    # quick helper function for Bernoulli trials - returns 0 or 1
    return 1 if (np.random.rand() < p) else 0

def run_ucb(T, means, version='original', nonstationary=False):
    """
    Running the bandit algorithm.
    - version='original' uses the original bonus term; any other value uses a modified bonus.
    - If nonstationary is True, then for arm 1, after t >= T/2 the reward probability is forced to 0.
    
    means = probabilities for each arm
    returns the cumulative pseudo-regret over time.
    """
    n_arms = len(means)
    best_arm = np.argmax(means)
    gap = [means[best_arm] - means[a] for a in range(n_arms)]
    
    counts = np.zeros(n_arms, dtype=int)  # pulls per arm
    values = np.zeros(n_arms)              # running average rewards
    regrets = np.zeros(T)
    cumulative_regret = 0.0
    
    # Pull each arm once
    for a in range(n_arms):
        # For nonstationary env, if arm==1 and t (a) >= T/2, use reward=bernoulli_sample(0)
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
        current_time = t + 1  # t starts at 0
        
        ucb_values = np.zeros(n_arms)
        for a in range(n_arms):
            avg_reward = values[a]
            if version == 'original':
                bonus = np.sqrt(1.5 * np.log(current_time) / counts[a])
            else:
                bonus = np.sqrt(np.log(current_time) / counts[a])
            ucb_values[a] = avg_reward + bonus
        
        chosen_arm = np.argmax(ucb_values)
        if nonstationary and chosen_arm == 1 and t >= T/2:
            reward = bernoulli_sample(0.0)
        else:
            reward = bernoulli_sample(means[chosen_arm])
        
        counts[chosen_arm] += 1
        values[chosen_arm] += (reward - values[chosen_arm]) / counts[chosen_arm]
        cumulative_regret += gap[chosen_arm]
        regrets[t] = cumulative_regret
    
    return regrets

def run_exp3(T, means, nonstationary=False):
    """
    EXP3 algorithm using the subtract-min trick to avoid overflow.
    We maintain a cumulative loss vector L for each arm.
    
    For rewards in [0,1] we define the loss as (1 - reward).
    If nonstationary is True, then for arm 1 after t >= T/2,
    the reward probability is forced to 0.
    
    :param T: time horizon
    :param means: list of reward probabilities for each arm
    :param nonstationary: if True, uses a nonstationary variant (breaks UCB)
    :return: cumulative pseudo-regret vector over time
    """
    n_arms = len(means)
    best_mean = np.max(means)  # best arm's mean (for regret computation)
    regrets = np.zeros(T)
    cumulative_regret = 0.0
    L = np.zeros(n_arms)  # cumulative losses for each arm
    
    for t in range(T):
        # time-varying learning rate; adjust as needed
        eta_t = np.sqrt(np.log(n_arms) / ((t+1) * n_arms))
        
        # Compute probabilities using the subtract-min trick:
        L_min = np.min(L)
        w = np.exp(-eta_t * (L - L_min))
        p = w / np.sum(w)
        
        chosen_arm = np.random.choice(n_arms, p=p)
        
        # In a nonstationary environment, force arm 1 to yield reward 0 after T/2.
        if nonstationary and chosen_arm == 1 and t >= T/2:
            reward = bernoulli_sample(0.0)
        else:
            reward = bernoulli_sample(means[chosen_arm])
        
        cumulative_regret += (best_mean - reward)
        regrets[t] = cumulative_regret
        
        # Estimated loss for the chosen arm: loss = (1 - reward) (since higher reward is better)
        # Then we use importance weighting:
        estimated_loss = (1 - reward) / p[chosen_arm] if p[chosen_arm] > 0 else 0
        L[chosen_arm] += estimated_loss

    return regrets

def run_experiment(T=100000, n_runs=20):
    """
    Existing experiment for different gaps (Delta) in an i.i.d. environment.
    It runs the two versions of UCB.
    """
    deltas = [1/4, 1/8, 1/16]  # different gap values
    fig, axes = plt.subplots(1, len(deltas), figsize=(18, 5))
    versions = ['original', 'modified']
    colors = ['red', 'blue']
    
    for idx, Delta in enumerate(deltas):
        mu_arm0 = 0.5 + 0.5 * Delta
        mu_arm1 = 0.5 - 0.5 * Delta
        means = [mu_arm0, mu_arm1]
        
        all_regrets = np.zeros((2, n_runs, T))
        for v, version in enumerate(versions):
            for r in range(n_runs):
                regrets = run_ucb(T, means, version=version, nonstationary=False)
                all_regrets[v, r, :] = regrets
        
        mean_regrets = np.mean(all_regrets, axis=1)
        std_regrets = np.std(all_regrets, axis=1)
        ax = axes[idx]
        x_vals = np.arange(1, T+1)
        for v, version in enumerate(versions):
            ax.plot(x_vals, mean_regrets[v], label=f"{version} UCB", color=colors[v])
            ax.fill_between(x_vals, mean_regrets[v] - std_regrets[v],
                            mean_regrets[v] + std_regrets[v],
                            color=colors[v], alpha=0.2)
        ax.set_title(f"Delta = {Delta}")
        ax.set_xlabel("t")
        ax.set_ylabel("Cumulative Pseudo-Regret")
        ax.legend()
    
    plt.tight_layout()
    plt.show()

def run_experiment_break(T=100000, n_runs=20):
    """
    New experiment (5.11 part b): nonstationary (adversarial) environment
    to "break" UCB1. Here, for two arms:
      - Arm 0: constant mean 0.6.
      - Arm 1: mean 0.9 for t < T/2, and 0.0 for t >= T/2.
    Compare UCB1 (original version) versus EXP3.
    """
    means = [0.6, 0.9]  # initial means for arms
    all_regrets_ucb = np.zeros((n_runs, T))
    all_regrets_exp3 = np.zeros((n_runs, T))
    
    for r in range(n_runs):
        regrets_ucb = run_ucb(T, means, version='original', nonstationary=True)
        all_regrets_ucb[r, :] = regrets_ucb
        regrets_exp3 = run_exp3(T, means, nonstationary=True)
        all_regrets_exp3[r, :] = regrets_exp3
    
    mean_regrets_ucb = np.mean(all_regrets_ucb, axis=0)
    std_regrets_ucb = np.std(all_regrets_ucb, axis=0)
    mean_regrets_exp3 = np.mean(all_regrets_exp3, axis=0)
    std_regrets_exp3 = np.std(all_regrets_exp3, axis=0)
    
    plt.figure(figsize=(6, 4))
    t_vals = np.arange(1, T+1)
    plt.plot(t_vals, mean_regrets_ucb, label="UCB1", color='blue')
    plt.fill_between(t_vals, mean_regrets_ucb - std_regrets_ucb,
                     mean_regrets_ucb + std_regrets_ucb, color='blue', alpha=0.2)
    plt.plot(t_vals, mean_regrets_exp3, label="EXP3", color='red')
    plt.fill_between(t_vals, mean_regrets_exp3 - std_regrets_exp3,
                     mean_regrets_exp3 + std_regrets_exp3, color='red', alpha=0.2)
    plt.xlabel("t")
    plt.ylabel("Cumulative Pseudo-Regret")
    plt.title("Nonstationary Environment (Breaking UCB1)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run the original experiment (i.i.d. case) for UCB only
    run_experiment(T=10000, n_runs=20)
    
    # Run the break experiment (nonstationary case) comparing UCB and EXP3
    run_experiment_break(T=10000, n_runs=20)
