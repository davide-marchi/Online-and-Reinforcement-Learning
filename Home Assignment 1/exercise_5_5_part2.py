import numpy as np
import matplotlib.pyplot as plt

def bernoulli_sample(p):
    # quick helper func for bernoulli trials - returns 0 or 1
    return 1 if (np.random.rand() < p) else 0

def run_ucb(T, means, version='original'):
    """
    Running the bandit stuff here. Two versions:
    - normal UCB1 (the one from lecture)
    - modified one from exercise 5.5 (slightly different bonus term)
    
    means = probabilities for each arm
    returns the regrets over time - need this for plotting later
    """
    n_arms = 2
    # figure out which arm is the best one
    best_arm = np.argmax(means)
    # calculate how much worse each arm is compared to the best one
    gap = [means[best_arm] - means[a] for a in range(n_arms)]
    
    # keeping track of stuff
    counts = np.zeros(n_arms, dtype=int)  # how many times we pulled each arm
    values = np.zeros(n_arms)  # running average rewards
    
    regrets = np.zeros(T)
    cumulative_regret = 0.0
    
    # gotta pull each arm once at the start (can't divide by zero)
    for a in range(n_arms):
        reward = bernoulli_sample(means[a])
        counts[a] = 1
        values[a] = reward
        cumulative_regret += gap[a]  # add up the regret
        regrets[a] = cumulative_regret
    
    # main loop - this is where the magic happens
    for t in range(n_arms, T):
        current_time = t + 1  # t starts at 0 but formulas need t >= 1
        
        # calculate UCB scores for each arm
        ucb_values = np.zeros(n_arms)
        for a in range(n_arms):
            avg_reward = values[a]
            # here's where the two versions differ:
            if version == 'original':
                # this is the one from class with sqrt(1.5 ln t / N)
                bonus = np.sqrt(1.5 * np.log(current_time) / counts[a])
            else:
                # modified version - just removed the 1.5 factor
                bonus = np.sqrt(np.log(current_time) / counts[a])
            ucb_values[a] = avg_reward + bonus
        
        # pick the arm with highest UCB score
        chosen_arm = np.argmax(ucb_values)
        
        # pull the arm and see what happens
        reward = bernoulli_sample(means[chosen_arm])
        
        # update our stats
        counts[chosen_arm] += 1
        values[chosen_arm] += (reward - values[chosen_arm]) / counts[chosen_arm]  # running average update
        
        # keep track of regret
        cumulative_regret += gap[chosen_arm]
        regrets[t] = cumulative_regret
    
    return regrets

def run_experiment(T=100000, n_runs=20):
    """
    testing both UCB versions with different gaps (Delta)
    running each combo multiple times to get averages
    """
    deltas = [1/4, 1/8, 1/16]  # different gaps to test
    
    # setting up the plots - one for each delta
    fig, axes = plt.subplots(1, len(deltas), figsize=(18, 5))
    
    # just to keep colors consistent
    versions = ['original', 'modified']
    colors = ['red', 'blue']
    
    for idx, Delta in enumerate(deltas):
        # set up the arms - arm0 is always the best one
        mu_arm0 = 0.5 + 0.5*Delta
        mu_arm1 = 0.5 - 0.5*Delta
        means = [mu_arm0, mu_arm1]
        
        # store results for all runs
        all_regrets = np.zeros((2, n_runs, T))
        
        for v, version in enumerate(versions):
            for r in range(n_runs):
                regrets = run_ucb(T, means, version=version)
                all_regrets[v, r, :] = regrets
        
        # calculate stats across runs
        mean_regrets = np.mean(all_regrets, axis=1)
        std_regrets = np.std(all_regrets, axis=1)
        
        # make it pretty
        ax = axes[idx]
        x_vals = np.arange(1, T+1)
        for v, version in enumerate(versions):
            ax.plot(x_vals, mean_regrets[v],
                    label=f"{version} UCB", color=colors[v])
            # add those nice shaded confidence intervals
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
    # let's run this thing
    run_experiment(T=100000, n_runs=20)
