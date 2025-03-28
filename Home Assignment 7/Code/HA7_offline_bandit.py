#!/usr/bin/env python3

import sys
import math
import matplotlib.pyplot as plt

def offline_ucb1_tracking(data, K, c=1.0):
    """
    Offline UCB1 with Importance Weighting.
    data: list of (arm, reward)
    K: number of arms
    Returns:
      - cumulative_rewards: list of cumulative reward at each round
      - average_rewards: list of average reward at each round
      - counts: number of updates per arm
    """
    REW = [0.0] * K  # cumulative scaled reward for each arm
    counts = [0] * K # update counts per arm
    cumulative_reward = 0.0
    cumulative_rewards = []
    average_rewards = []

    for t, (arm, reward) in enumerate(data, start=1):
        # Compute UCB index for each arm (reward maximization)
        ucb_values = []
        for i in range(K):
            if counts[i] == 0:
                ucb_values.append(float('inf'))
            else:
                avg = REW[i] / counts[i]
                bonus = c * math.sqrt(math.log(t) / counts[i])
                ucb_values.append(avg + bonus)
        # Choose arm with highest UCB index
        i_t = max(range(K), key=lambda i: ucb_values[i])
        # Update only if logging arm matches the policy's choice
        if arm == i_t:
            # importance weighting: multiply observed reward by K
            REW[arm] += K * reward
            counts[arm] += 1
            cumulative_reward += reward
        cumulative_rewards.append(cumulative_reward)
        average_rewards.append(cumulative_reward / t)
    return cumulative_rewards, average_rewards, counts

def offline_exp3_tracking(data, K):
    """
    Offline EXP3 (Anytime) with Importance Weighting.
    data: list of (arm, reward)
    K: number of arms
    Returns:
      - cumulative_rewards: list of cumulative reward at each round
      - average_rewards: list of average reward at each round
      - counts: number of updates per arm (when logging arm matches policy's draw)
    """
    L = [0.0] * K         # cumulative importance-weighted losses
    counts = [0] * K      # update counts per arm
    cumulative_reward = 0.0
    cumulative_rewards = []
    average_rewards = []
    
    for t, (arm, reward) in enumerate(data, start=1):
        # Compute distribution p_t with numerical stability:
        offset = max(-L[j] for j in range(K))
        denom = sum(math.exp(-L[j] + offset) for j in range(K))
        if denom == 0:
            denom = 1e-10
        p = [math.exp(-L[i] + offset) / denom for i in range(K)]
        
        # For an offline replay, we assume that if the logging arm equals the arm 
        # with highest probability, we update.
        chosen_by_exp3 = max(range(K), key=lambda i: p[i])
        # Anytime learning rate (if needed later):
        eta_t = math.sqrt((2.0 * math.log(K)) / (K * t))
        
        if arm == chosen_by_exp3:
            # Define loss = 1 - reward (so that high reward gives low loss)
            loss = 1.0 - reward
            # Importance weight is K, since logging probability is 1/K
            L[arm] += K * loss
            counts[arm] += 1
            cumulative_reward += reward
        cumulative_rewards.append(cumulative_reward)
        average_rewards.append(cumulative_reward / t)
    return cumulative_rewards, average_rewards, counts

def read_data(file_path):
    """
    Reads the dataset file.
    Each line is expected to have 12 space-separated values:
      <chosen_arm> <f1> ... <f10> <reward>
    We only use the first and the last columns.
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            cols = line.strip().split()
            if len(cols) < 2:
                continue
            arm = int(cols[0])
            reward = float(cols[-1])
            data.append((arm, reward))
    return data

def plot_results(ucb1_cum, ucb1_avg, exp3_cum, exp3_avg, rounds, ucb1_counts, exp3_counts, K):
    # Plot 1: Cumulative Reward over Time
    plt.figure(figsize=(8, 6))
    plt.plot(rounds, ucb1_cum, label="Offline UCB1")
    plt.plot(rounds, exp3_cum, label="Offline EXP3")
    plt.xlabel("Round")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward vs. Round")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 2: Average Reward over Time
    plt.figure(figsize=(8, 6))
    plt.plot(rounds, ucb1_avg, label="Offline UCB1")
    plt.plot(rounds, exp3_avg, label="Offline EXP3")
    plt.xlabel("Round")
    plt.ylabel("Average Reward")
    plt.title("Average Reward vs. Round")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 3: Number of Updates per Arm (Bar Plot)
    arms = list(range(K))
    width = 0.35  # width of the bars
    plt.figure(figsize=(8, 6))
    plt.bar([a - width/2 for a in arms], ucb1_counts, width, label="UCB1")
    plt.bar([a + width/2 for a in arms], exp3_counts, width, label="EXP3")
    plt.xlabel("Arm")
    plt.ylabel("Update Count")
    plt.title("Number of Updates per Arm")
    plt.xticks(arms)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():

    file_path = "./Home Assignment 7/Code/data_preprocessed_features"
    data = read_data(file_path)
    if not data:
        print("No data read from file.")
        sys.exit(1)
    # Determine K (assuming arms are numbered starting at 0)
    max_arm = max(arm for arm, _ in data)
    K = max_arm + 1
    print(f"Loaded {len(data)} rounds with arms in [0, {K-1}].")

    rounds = list(range(1, len(data) + 1))
    
    # Run offline UCB1 and record performance
    ucb1_cum, ucb1_avg, ucb1_counts = offline_ucb1_tracking(data, K, c=1.0)
    # Run offline EXP3 and record performance
    exp3_cum, exp3_avg, exp3_counts = offline_exp3_tracking(data, K)
    
    print(f"[Offline UCB1] Final cumulative reward: {ucb1_cum[-1]:.3f}, Average reward: {ucb1_avg[-1]:.4f}")
    print(f"[Offline EXP3] Final cumulative reward: {exp3_cum[-1]:.3f}, Average reward: {exp3_avg[-1]:.4f}")
    print(f"UCB1 update counts per arm: {ucb1_counts}")
    print(f"EXP3 update counts per arm: {exp3_counts}")

    # Generate the requested plots
    plot_results(ucb1_cum, ucb1_avg, exp3_cum, exp3_avg, rounds, ucb1_counts, exp3_counts, K)

if __name__ == "__main__":
    main()
