import math

def offline_ucb1(data, K, c=1.0):
    """
    Offline UCB1 with Importance Weighting.
    data: list of (arm, reward)
    K: number of arms
    c: confidence parameter
    Returns total reward, plus arrays for per-arm stats.
    """
    # REW[i] = cumulative scaled reward for arm i
    # N[i]   = number of times arm i was effectively chosen
    REW = [0.0]*K
    N   = [0]*K

    total_reward = 0.0

    for t, (arm, reward) in enumerate(data, start=1):
        # 1) Compute UCB indices (assuming reward maximization)
        ucb_values = []
        for i in range(K):
            if N[i] == 0:
                ucb_values.append(float('inf'))
            else:
                avg = REW[i] / N[i]
                bonus = c * math.sqrt(math.log(t) / N[i])
                ucb_values.append(avg + bonus)

        # 2) UCB1 picks arm i_t = argmax of UCB indices
        i_t = max(range(K), key=lambda i: ucb_values[i])

        # 3) Update if the logging policy's chosen arm == i_t
        if arm == i_t:
            # Importance weight is K, because logging is uniform
            REW[arm] += K * reward
            N[arm]   += 1
            total_reward += reward

    return total_reward, REW, N

def offline_exp3_anytime(data, K):
    """
    Offline EXP3 (Anytime) with Importance Weighting.
    data: list of (arm, reward)
    K: number of arms
    Returns total reward, plus the final 'loss' array.
    """
    # We'll treat the data as reward in [0,1], and define loss = 1 - reward.
    # L[i] will accumulate the importance-weighted losses for arm i.
    L = [0.0]*K
    total_reward = 0.0

    for t, (arm, reward) in enumerate(data, start=1):
        # Compute distribution p_t with numerical stability
        offset = max(-L[j] for j in range(K))
        denom = sum(math.exp(-L[j] + offset) for j in range(K))
        if denom == 0:
            denom = 1e-10
        p = [math.exp(-L[i] + offset) / denom for i in range(K)]

        # 2) Anytime learning rate
        eta_t = math.sqrt((2.0 * math.log(K)) / (K * t))

        # 3) We'll do a simple approach: if arm == argmax of p, we treat that as a match
        chosen_by_exp3 = max(range(K), key=lambda i: p[i])
        if arm == chosen_by_exp3:
            # Loss = 1 - reward, scaled by K
            loss = 1.0 - reward
            L[arm] += K * loss
            total_reward += reward

        # Next iteration, p will be recomputed from L, so no direct w-update is needed

    return total_reward, L

def main():
    # File path given directly here for simplicity:
    file_path = "./Home Assignment 7/Code/data_preprocessed_features"

    # Read data lines. Each line has 12 columns:
    #   <arm> <f1> ... <f10> <reward>
    # We'll keep only the first (arm) and the last (reward).
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            cols = line.strip().split()
            # e.g., cols might be ["14","0","1","1","0","0","1","0","0","1","0","1"]
            # arm = int(cols[0])        # chosen arm
            # reward = float(cols[-1])  # reward
            arm = int(cols[0])
            reward = float(cols[-1])
            data.append((arm, reward))

    # Determine K by looking at max arm
    max_arm = max(arm for arm, _ in data)
    K = max_arm + 1

    # 1) Offline UCB1
    ucb1_reward, ucb1_REW, ucb1_N = offline_ucb1(data, K)
    avg_ucb1 = ucb1_reward / len(data)
    print(f"[Offline UCB1] Total reward: {ucb1_reward:.3f}, Average reward: {avg_ucb1:.4f}")

    # 2) Offline EXP3 (Anytime)
    exp3_reward, exp3_losses = offline_exp3_anytime(data, K)
    avg_exp3 = exp3_reward / len(data)
    print(f"[Offline EXP3] Total reward: {exp3_reward:.3f}, Average reward: {avg_exp3:.4f}")

if __name__ == "__main__":
    main()
