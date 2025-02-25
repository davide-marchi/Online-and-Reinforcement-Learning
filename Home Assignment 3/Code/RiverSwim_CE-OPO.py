import numpy as np
import matplotlib.pyplot as plt
import types

# --- The given RiverSwim environment ---
class riverswim():
    def __init__(self, nS):
        self.nS = nS
        self.nA = 2
        # Build the transition matrix P and support lists.
        self.P = np.zeros((nS, 2, nS))
        self.support = [[[] for _ in range(self.nA)] for _ in range(nS)]
        for s in range(nS):
            if s == 0:
                self.P[s, 0, s] = 1
                self.P[s, 1, s] = 0.6
                self.P[s, 1, s + 1] = 0.4
                self.support[s][0].append(0)
                self.support[s][1] += [0, 1]
            elif s == nS - 1:
                self.P[s, 0, s - 1] = 1
                self.P[s, 1, s] = 0.6
                self.P[s, 1, s - 1] = 0.4
                self.support[s][0].append(s - 1)
                self.support[s][1] += [s - 1, s]
            else:
                self.P[s, 0, s - 1] = 1
                self.P[s, 1, s] = 0.55
                self.P[s, 1, s + 1] = 0.4
                self.P[s, 1, s - 1] = 0.05
                self.support[s][0].append(s - 1)
                self.support[s][1] += [s - 1, s, s + 1]
        # Reward matrix: only state 0 (leftmost) and state nS-1 (rightmost) have nonzero rewards.
        self.R = np.zeros((nS, 2))
        self.R[0, 0] = 0.05   # when at leftmost state, action 'left'
        self.R[nS - 1, 1] = 1 # when at rightmost state, action 'right'
        self.s = 0

    def reset(self):
        self.s = 0
        return self.s

    def step(self, action):
        new_s = np.random.choice(np.arange(self.nS), p=self.P[self.s, action])
        reward = self.R[self.s, action]
        self.s = new_s
        return new_s, reward

# --- Value Iteration to compute Q-values ---
def value_iteration(P, R, gamma=0.98, theta=1e-8, max_iter=10000):
    nS, nA, _ = P.shape
    V = np.zeros(nS)
    for i in range(max_iter):
        V_prev = V.copy()
        for s in range(nS):
            Q_sa = np.zeros(nA)
            for a in range(nA):
                Q_sa[a] = R[s, a] + gamma * np.dot(P[s, a], V_prev)
            V[s] = np.max(Q_sa)
        if np.max(np.abs(V - V_prev)) < theta:
            break
    Q = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            Q[s, a] = R[s, a] + gamma * np.dot(P[s, a], V)
    return V, Q

# --- CE-OPO algorithm ---
def ce_opo(env, true_P, true_R, gamma=0.98, epsilon=0.15, alpha_smooth=0.1,
           horizon=int(1e6), eval_interval=10000):
    nS = env.nS
    nA = env.nA

    # Initialize counts and reward accumulators.
    N_sa = np.zeros((nS, nA))
    N_sas = np.zeros((nS, nA, nS))
    R_sum = np.zeros((nS, nA))

    # Initialize Q-estimate arbitrarily.
    Q_est = np.zeros((nS, nA))

    # Precompute true Q-values using the known true model.
    _, Q_true = value_iteration(true_P, true_R, gamma)
    pi_true = np.argmax(Q_true, axis=1)  # True optimal policy

    # Lists to record performance metrics.
    steps = []
    Q_diff_list = []
    policy_diff_list = []
    return_loss_list = []

    s = env.reset()
    for t in range(1, horizon + 1):
        # ε-greedy action selection using current Q_est.
        if np.random.rand() < epsilon:
            a = np.random.randint(nA)
        else:
            a = np.argmax(Q_est[s])
        # Take action and observe transition.
        s_next, r = env.step(a)
        # Update counts.
        N_sa[s, a] += 1
        N_sas[s, a, s_next] += 1
        R_sum[s, a] += r
        s = s_next

        # Update model and evaluate metrics every 'eval_interval' steps.
        if t % eval_interval == 0:
            P_est = np.zeros((nS, nA, nS))
            R_est = np.zeros((nS, nA))
            for s_i in range(nS):
                for a_i in range(nA):
                    denom = N_sa[s_i, a_i] + alpha_smooth * nS
                    if denom == 0:
                        P_est[s_i, a_i] = np.ones(nS) / nS
                        R_est[s_i, a_i] = 0
                    else:
                        P_est[s_i, a_i] = (N_sas[s_i, a_i] + alpha_smooth) / denom
                        R_est[s_i, a_i] = (R_sum[s_i, a_i] + alpha_smooth) / (N_sa[s_i, a_i] + alpha_smooth)
            # Solve the estimated MDP.
            _, Q_est = value_iteration(P_est, R_est, gamma)
            pi_est = np.argmax(Q_est, axis=1)
            Q_diff = np.max(np.abs(Q_true - Q_est))
            policy_diff = np.sum(pi_est != pi_true)
            # We define the return loss at the left bank (state 0) for action 'right' (action index 1).
            return_loss = Q_true[0, 1] - Q_est[0, 1]
            steps.append(t)
            Q_diff_list.append(Q_diff)
            policy_diff_list.append(policy_diff)
            return_loss_list.append(return_loss)
            if t % (10 * eval_interval) == 0:
                print(f"Step {t}: Q_diff = {Q_diff:.4f}, PolicyDiff = {policy_diff}, ReturnLoss = {return_loss:.4f}")

    results = {
        "steps": steps,
        "Q_diff": Q_diff_list,
        "policy_diff": policy_diff_list,
        "return_loss": return_loss_list,
        "pi_true": pi_true,
        "pi_est": pi_est,
        "Q_true": Q_true,
        "Q_est": Q_est
    }
    return results

# --- Main Experiment: Original RiverSwim MDP ---
nS = 4
env = riverswim(nS)
true_P = env.P
true_R = env.R.copy()

print("Running CE-OPO on the original RiverSwim MDP...")
results_original = ce_opo(env, true_P, true_R, gamma=0.98, epsilon=0.15,
                           alpha_smooth=0.1, horizon=int(1e6), eval_interval=1000)

# Plot the performance metrics.
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(results_original["steps"], results_original["Q_diff"])
plt.xlabel('Time steps')
plt.ylabel(r'$\|Q^*-Q_t\|_\infty$')
plt.title('Q-value Error')

plt.subplot(1,3,2)
plt.plot(results_original["steps"], results_original["policy_diff"])
plt.xlabel('Time steps')
plt.ylabel('Policy Diff')
plt.title('Policy Mismatch Count')

plt.subplot(1,3,3)
plt.plot(results_original["steps"], results_original["return_loss"])
plt.xlabel('Time steps')
plt.ylabel('Return Loss')
plt.title('Return Loss at state 0 (action right)')

plt.tight_layout()
plt.savefig('Home Assignment 3/Code/RiverSwim_CE-OPO.png')
plt.show()

# --- Experiment for the Variant with Random Reward in State 0 (right) ---
# Create a new environment instance.
env_variant = riverswim(nS)
# Modify the step function so that when the agent is in state 0 and takes action 'right' (a==1),
# the reward is drawn uniformly from [0,2].
def variant_step(self, action):
    # If in the rightmost state (nS-1) and taking action 'right' (1):
    if self.s == self.nS - 1 and action == 1:
        reward = np.random.uniform(0, 2)
    else:
        reward = self.R[self.s, action]
    new_s = np.random.choice(np.arange(self.nS), p=self.P[self.s, action])
    self.s = new_s
    return new_s, reward

env_variant.step = types.MethodType(variant_step, env_variant)
env_variant.reset()
true_R_variant = env_variant.R.copy()
# For evaluation, the expected reward at the rightmost state (nS-1) for action 1 is 1.
true_R_variant[nS - 1, 1] = 1

print("Running CE-OPO on the variant RiverSwim MDP (random reward at state nS-1, action right)...")
results_variant = ce_opo(env_variant, true_P, true_R_variant, gamma=0.98,
                          epsilon=0.15, alpha_smooth=0.1, horizon=int(1e6), eval_interval=1000)

# Plot the performance metrics for the variant.
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(results_variant["steps"], results_variant["Q_diff"])
plt.xlabel('Time steps')
plt.ylabel(r'$\|Q^*-Q_t\|_\infty$')
plt.title('Q-value Error (Variant)')

plt.subplot(1,3,2)
plt.plot(results_variant["steps"], results_variant["policy_diff"])
plt.xlabel('Time steps')
plt.ylabel('Policy Diff')
plt.title('Policy Mismatch Count (Variant)')

plt.subplot(1,3,3)
plt.plot(results_variant["steps"], results_variant["return_loss"])
plt.xlabel('Time steps')
plt.ylabel('Return Loss')
plt.title('Return Loss at state 0 (action right, Variant)')

plt.tight_layout()
plt.savefig('Home Assignment 3/Code/RiverSwim_CE-OPO_Variant.png')
plt.show()
