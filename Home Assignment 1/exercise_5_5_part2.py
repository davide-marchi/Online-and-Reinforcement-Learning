import numpy as np
import matplotlib.pyplot as plt

def bernoulli_sample(p):
    """Draw a Bernoulli( p ) sample, returning 0 or 1."""
    return 1 if (np.random.rand() < p) else 0

def run_ucb(T, means, version='original'):
    """
    Run one trial of a two-armed bandit for T steps using either:
      - version='original': the standard UCB1 from Section 5.3
      - version='modified': the improved UCB from Exercise 5.5
    means = [mean_arm0, mean_arm1] are the Bernoulli parameters for arms 0 and 1.
    
    Returns:
      regrets: array of length T, where regrets[t] = cumulative pseudo-regret at time t+1
               (i.e. sum_{s=1..t+1} Delta(A_s), using Delta(arm) = mu(a^*) - mu(arm)).
    """
    n_arms = 2
    # Identify the best arm by maximum mean
    best_arm = np.argmax(means)
    # The gap Delta(a) = mu(best_arm) - mu(a)
    gap = [means[best_arm] - means[a] for a in range(n_arms)]
    
    # Tracking counts and average rewards
    counts = np.zeros(n_arms, dtype=int)
    values = np.zeros(n_arms)  # empirical mean rewards
    
    regrets = np.zeros(T)
    cumulative_regret = 0.0
    
    # --- Initialization: play each arm once to get non-zero counts ---
    for a in range(n_arms):
        reward = bernoulli_sample(means[a])
        counts[a] = 1
        values[a] = reward
        # Pseudo-regret update
        cumulative_regret += gap[a]
        regrets[a] = cumulative_regret
    
    # Main loop
    for t in range(n_arms, T):
        # Current time index is t+1 in 1-based
        current_time = t + 1
        
        # Compute UCB indices
        ucb_values = np.zeros(n_arms)
        for a in range(n_arms):
            # Empirical mean
            avg_reward = values[a]
            # Confidence radius depends on version
            if version == 'original':
                # E.g. standard UCB1 from sec.5.3 with sqrt(1.5 ln t / N).
                bonus = np.sqrt(1.5 * np.log(current_time) / counts[a])
            else:
                # 'modified' version from exercise 5.5, with sqrt(ln t / N)
                bonus = np.sqrt(np.log(current_time) / counts[a])
            ucb_values[a] = avg_reward + bonus
        
        # Choose the arm that maximizes the UCB index
        chosen_arm = np.argmax(ucb_values)
        
        # Get the Bernoulli reward
        reward = bernoulli_sample(means[chosen_arm])
        
        # Update counts and values
        counts[chosen_arm] += 1
        # Incremental update of average
        values[chosen_arm] += (reward - values[chosen_arm]) / counts[chosen_arm]
        
        # Update pseudo-regret
        cumulative_regret += gap[chosen_arm]
        regrets[t] = cumulative_regret
    
    return regrets

def run_experiment(T=100000, n_runs=20):
    """
    Runs the experiment for Delta in {1/4, 1/8, 1/16}, both versions of UCB,
    each repeated n_runs times, and produces comparison plots.
    """
    deltas = [1/4, 1/8, 1/16]
    
    # Prepare figure with one subplot per Delta
    fig, axes = plt.subplots(1, len(deltas), figsize=(18, 5))
    
    # For consistent line colors/symbols
    versions = ['original', 'modified']
    colors = ['red', 'blue']
    
    for idx, Delta in enumerate(deltas):
        # Means for the two arms: arm0 is best
        mu_arm0 = 0.5 + 0.5*Delta  # best arm
        mu_arm1 = 0.5 - 0.5*Delta
        means = [mu_arm0, mu_arm1]
        
        # We'll store regrets across runs, for each version
        # shape = (2 versions, n_runs, T)
        all_regrets = np.zeros((2, n_runs, T))
        
        for v, version in enumerate(versions):
            for r in range(n_runs):
                regrets = run_ucb(T, means, version=version)
                all_regrets[v, r, :] = regrets
        
        # Compute mean and std over runs
        mean_regrets = np.mean(all_regrets, axis=1)
        std_regrets = np.std(all_regrets, axis=1)
        
        # Plot
        ax = axes[idx]
        x_vals = np.arange(1, T+1)
        for v, version in enumerate(versions):
            ax.plot(x_vals, mean_regrets[v],
                    label=f"{version} UCB", color=colors[v])
            # Shaded area for +/- 1 std
            ax.fill_between(x_vals,
                            mean_regrets[v] - std_regrets[v],
                            mean_regrets[v] + std_regrets[v],
                            color=colors[v], alpha=0.2)
        
        ax.set_title(f"Delta = {Delta}")
        ax.set_xlabel("t")
        ax.set_ylabel("Cumulative Pseudo-Regret")
        ax.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run everything in one go
    run_experiment(T=100000, n_runs=20)
