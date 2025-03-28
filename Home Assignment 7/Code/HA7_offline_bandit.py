import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# 1) DATA LOADING
###############################################################################
def load_data(file_path):
    """
    Loads the logged bandit data from file_path.

    The file is assumed to have whitespace-separated values with 12 columns.
    - Column 0: the arm chosen by the logging policy (integer, e.g., in 0..K-1)
    - Column -1: the reward (binary: 0 or 1)
    - Columns 1 to 10: additional features (unused here)

    Returns:
      A: 1D numpy array of shape (T,) with the logged arm for each round.
      R: 1D numpy array of shape (T,) with the reward for that round.
      K: number of arms (computed as max(A)+1)
      T: total number of rounds (number of rows in the file)
    """
    # Load all data; np.loadtxt uses any whitespace as delimiter by default.
    data = np.loadtxt(file_path)
    T = data.shape[0]
    
    # Extract the logged arm (first column) and reward (last column)
    A = data[:, 0].astype(int)
    R = data[:, -1]
    
    # Determine number of arms; assumes arms are labeled 0,...,K-1.
    K = int(np.max(A)) + 1

    print(f"Data loaded from {file_path}: {T} rounds, {K} arms.")
    return A, R, K, T

###############################################################################
# 2) OFFLINE UCB1 (IMPORTANCE-WEIGHTED)
###############################################################################
def offline_ucb1(A, R, K, c=2.0):
    """
    Offline UCB1 with importance weighting for uniform logging.

    Updates occur only when the algorithm’s chosen arm matches the logged arm.
    The observed reward is scaled by the importance weight K.

    Returns:
      cumulative_rewards: 1D array of shape (T,) with cumulative reward.
    """
    T = len(A)
    L = np.zeros(K)         # Sum of (importance-weighted reward) per arm.
    N = np.zeros(K, dtype=int)  # Number of updates per arm.

    cumulative_rewards = np.zeros(T)

    for t in range(T):
        # Compute UCB index for each arm
        ucb_values = np.zeros(K)
        for i in range(K):
            if N[i] == 0:
                ucb_values[i] = np.inf
            else:
                avg_reward = L[i] / N[i]  # Here L[i] is sum of (K*reward)
                # Note: We use UCB for minimization if treating reward as "loss"
                # but here we simply add an exploration bonus:
                ucb_values[i] = avg_reward + c * np.sqrt(np.log(t+1) / N[i])
        
        # Choose the arm with the minimum UCB value (or maximum, if you invert the sign)
        # Here we follow the lecture notes structure.
        it = np.argmin(ucb_values)

        # Only update if the logging policy picked the same arm
        if A[t] == it:
            iw_reward = K * R[t]  # importance weighting because logging is uniform
            L[it] += iw_reward
            N[it] += 1
            cumulative_rewards[t] = iw_reward if t == 0 else cumulative_rewards[t-1] + iw_reward
        else:
            cumulative_rewards[t] = cumulative_rewards[t-1] if t > 0 else 0.0

    return cumulative_rewards

###############################################################################
# 3) OFFLINE EXP3 (ANYTIME VERSION, IMPORTANCE-WEIGHTED)
###############################################################################
def offline_exp3(A, R, K):
    """
    Offline EXP3 (anytime version) with importance weighting.

    Uses eta_t = sqrt(2 ln(K)/(K * t)). The algorithm updates only when the logging
    policy's chosen arm is used.
    
    Returns:
      cumulative_rewards: 1D array of shape (T,) with cumulative reward.
    """
    T = len(A)
    Ltilde = np.zeros(K, dtype=float)  # Cumulative importance-weighted loss (or reward aggregator)
    cumulative_rewards = np.zeros(T)

    for t in range(T):
        # Compute learning rate eta_t
        eta_t = np.sqrt((2.0 * np.log(K)) / (K * (t+1)))
        
        # Compute weights and probability distribution p_t
        w = np.exp(-eta_t * Ltilde)
        W = np.sum(w)
        p = w / W

        # In offline evaluation we update using the logged arm.
        chosen_arm = A[t]
        iw_reward = K * R[t]
        # Update: the lecture notes add the scaled loss; here we assume reward is the "loss" term.
        # (If your loss were defined as 1 - reward, adjust accordingly.)
        Ltilde[chosen_arm] += K * R[t]
        
        cumulative_rewards[t] = iw_reward if t == 0 else cumulative_rewards[t-1] + iw_reward

    return cumulative_rewards

###############################################################################
# 4) BASELINE: ALWAYS PICK CERTAIN ARM(S)
###############################################################################
def offline_fixed_subset(A, R, subset_of_arms, K):
    """
    Evaluates the policy that always picks an arm in the given subset.
    Returns a cumulative reward array computed via importance weighting.
    """
    T = len(A)
    subset_of_arms = set(subset_of_arms)
    cumr = np.zeros(T)
    for t in range(T):
        iw_reward = K * R[t] if A[t] in subset_of_arms else 0.0
        cumr[t] = iw_reward if t == 0 else cumr[t-1] + iw_reward
    return cumr

###############################################################################
# 5) PLOTTING & MAIN SCRIPT
###############################################################################
def main():
    # A) Load the logged data.
    file_path = "./Home Assignment 7/Code/data_preprocessed_features"
    A, R, K, T = load_data(file_path)

    # B) Run offline evaluation for UCB1 and EXP3.
    print("Running Offline UCB1 ...")
    cumr_ucb1 = offline_ucb1(A, R, K, c=2.0)

    print("Running Offline EXP3 ...")
    cumr_exp3 = offline_exp3(A, R, K)

    # C) Build baseline fixed-arm policies by computing empirical average rewards.
    counts = np.zeros(K, dtype=int)
    sums   = np.zeros(K, dtype=float)
    for t in range(T):
        counts[A[t]] += 1
        sums[A[t]]   += R[t]
    avg_rewards = np.array([sums[i] / counts[i] if counts[i] > 0 else 0.0 for i in range(K)])
    
    # Sort arms by average reward (best to worst)
    sorted_arms = np.argsort(-avg_rewards)
    best_arm    = [sorted_arms[0]]
    worst_arm   = [sorted_arms[-1]]
    two_worst   = [sorted_arms[-1], sorted_arms[-2]]
    three_worst = [sorted_arms[-1], sorted_arms[-2], sorted_arms[-3]]
    median_arm  = [sorted_arms[K // 2]]

    # Baseline cumulative reward curves.
    cumr_best_arm    = offline_fixed_subset(A, R, best_arm, K)
    cumr_worst_arm   = offline_fixed_subset(A, R, worst_arm, K)
    cumr_two_worst   = offline_fixed_subset(A, R, two_worst, K)
    cumr_three_worst = offline_fixed_subset(A, R, three_worst, K)
    cumr_median_arm  = offline_fixed_subset(A, R, median_arm, K)

    # D) Compute an optional theoretical bound for EXP3.
    bound_exp3 = np.zeros(T)
    const = np.sqrt(2.0 * K * np.log(K))
    for t in range(T):
        bound_exp3[t] = const * np.sqrt(t+1)

    # E) Plot the curves.
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.ravel()

    # (a) Plot all single-arm curves with UCB1, EXP3 and the bound.
    ax = axes[0]
    for i in range(K):
        cumr_arm_i = offline_fixed_subset(A, R, [i], K)
        ax.plot(cumr_arm_i, alpha=0.2)
    ax.plot(cumr_ucb1, label="UCB1", lw=2)
    ax.plot(cumr_exp3, label="EXP3", lw=2)
    ax.plot(bound_exp3, "--", label="EXP3 bound", color='orange')
    ax.set_title("(a) All arms")
    ax.set_xlabel("t")
    ax.set_ylabel("Cumulative Reward")
    ax.legend()

    # (b) Best vs. worst arm.
    ax = axes[1]
    ax.plot(cumr_best_arm,  label="Best arm", color='blue')
    ax.plot(cumr_worst_arm, label="Worst arm", color='red')
    ax.plot(cumr_ucb1,      label="UCB1", color='green')
    ax.plot(cumr_exp3,      label="EXP3", color='magenta')
    ax.plot(bound_exp3,     "--", label="EXP3 bound", color='orange')
    ax.set_title("(b) Best, worst action")
    ax.set_xlabel("t")
    ax.set_ylabel("Cumulative Reward")
    ax.legend()

    # (c) Best vs. two worst arms.
    ax = axes[2]
    ax.plot(cumr_best_arm,   label="Best arm", color='blue')
    ax.plot(cumr_two_worst,  label="2 worst arms", color='red')
    ax.plot(cumr_ucb1,       label="UCB1", color='green')
    ax.plot(cumr_exp3,       label="EXP3", color='magenta')
    ax.plot(bound_exp3,      "--", label="EXP3 bound", color='orange')
    ax.set_title("(c) Best, two worst actions")
    ax.set_xlabel("t")
    ax.set_ylabel("Cumulative Reward")
    ax.legend()

    # (d) Best vs. three worst arms.
    ax = axes[3]
    ax.plot(cumr_best_arm,   label="Best arm", color='blue')
    ax.plot(cumr_three_worst, label="3 worst arms", color='red')
    ax.plot(cumr_ucb1,       label="UCB1", color='green')
    ax.plot(cumr_exp3,       label="EXP3", color='magenta')
    ax.plot(bound_exp3,      "--", label="EXP3 bound", color='orange')
    ax.set_title("(d) Best, three worst actions")
    ax.set_xlabel("t")
    ax.set_ylabel("Cumulative Reward")
    ax.legend()

    # (e) Best, median, worst arms.
    ax = axes[4]
    ax.plot(cumr_best_arm,    label="Best arm", color='blue')
    ax.plot(cumr_median_arm,  label="Median arm", color='gray')
    ax.plot(cumr_worst_arm,   label="Worst arm", color='red')
    ax.plot(cumr_ucb1,        label="UCB1", color='green')
    ax.plot(cumr_exp3,        label="EXP3", color='magenta')
    ax.plot(bound_exp3,       "--", label="EXP3 bound", color='orange')
    ax.set_title("(e) Best, median, worst action")
    ax.set_xlabel("t")
    ax.set_ylabel("Cumulative Reward")
    ax.legend()

    # Hide the unused subplot.
    axes[5].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
