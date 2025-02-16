# Written by Hippolyte Bourel
# This code is proposed as a reference solution for various exercises of Home Assignements for the OReL course in 2024.
# This solution is tailored for simplicity of understanding and is in no way optimal, nor the only way to implement the different elements!
import numpy as np

# A simple riverswim implementation with chosen number of state 'nS' chosen in input.
# We arbitrarily chose the action '0' = 'go to the left' thus '1' = 'go to the right'.
# Finally the state '0' is the leftmost, 'nS - 1' is the rightmost.
class riverswim():

	def __init__(self, nS):
		self.nS = nS
		self.nA = 2

		# We build the transitions matrix P, and its associated support lists.
		self.P = np.zeros((nS, 2, nS))
		self.support = [[[] for _ in range(self.nA)] for _ in range(self.nS)]
		for s in range(nS):
			if s == 0:
				self.P[s, 0, s] = 1
				self.P[s, 1, s] = 0.6
				self.P[s, 1, s + 1] = 0.4
				self.support[s][0] += [0]
				self.support[s][1] += [0, 1]
			elif s == nS - 1:
				self.P[s, 0, s - 1] = 1
				self.P[s, 1, s] = 0.6
				self.P[s, 1, s - 1] = 0.4
				self.support[s][0] += [s - 1]
				self.support[s][1] += [s - 1, s]
			else:
				self.P[s, 0, s - 1] = 1
				self.P[s, 1, s] = 0.55
				self.P[s, 1, s + 1] = 0.4
				self.P[s, 1, s - 1] = 0.05
				self.support[s][0] += [s - 1]
				self.support[s][1] += [s - 1, s, s + 1]
		
		# We build the reward matrix R.
		self.R = np.zeros((nS, 2))
		self.R[0, 0] = 0.05
		self.R[nS - 1, 1] = 1

		# We (arbitrarily) set the initial state in the leftmost position.
		self.s = 0

	# To reset the environment in initial settings.
	def reset(self):
		self.s = 0
		return self.s

	# Perform a step in the environment for a given action. Return a couple state, reward (s_t, r_t).
	def step(self, action):
		new_s = np.random.choice(np.arange(self.nS), p=self.P[self.s, action])
		reward = self.R[self.s, action]
		self.s = new_s
		return new_s, reward
	

###########################################################################
# 1) Monte Carlo approximation of V^pi
###########################################################################

# --- USER PARAMETERS ---
gamma = 0.96         # discount factor
num_episodes = 50    # number of trajectories for each initial state
T = 300              # length of each trajectory

# We define the policy pi(s):
#  - for states s in [0,1,2], go RIGHT with prob 0.65 and LEFT with prob 0.35
#  - for states s in [3,4], always go RIGHT (prob = 1).
def sample_action_from_policy(s):
    """
    Returns 0 for 'left' or 1 for 'right', 
    according to the desired policy pi(s).
    """
    if s <= 2:
        # with prob 0.65 pick 'right' (action=1), else 'left' (action=0)
        return np.random.choice([0,1], p=[0.35,0.65])
    else:
        # states 4 or 5 -> always right
        return 1

# Create the environment
env = riverswim(5)

# This array will store our Monte Carlo estimate of V^pi(s)
Vhat_MC = np.zeros(env.nS)

# Loop over every state s as the "start state"
for s in range(env.nS):
    returns_sum = 0.0

    # We'll simulate 'num_episodes' trajectories from each state s
    for _ in range(num_episodes):
        # Manually set the env's current state to s
        env.s = s
        
        G = 0.0      # discounted return
        discount = 1.0
        for t in range(T):
            # Sample action from our policy
            a = sample_action_from_policy(env.s)
            s_next, r = env.step(a)
            
            G += discount * r
            discount *= gamma
            # proceed to next state
            # env.s is already updated to s_next
        returns_sum += G

    # Average over all episodes
    Vhat_MC[s] = returns_sum / num_episodes

print("Monte Carlo estimates of V^pi(s) for s=0..4:")
for s in range(env.nS):
    print(f"State {s}: {Vhat_MC[s]:.5f}")
print()

###########################################################################
# 2) Exact value of V^pi by direct computation
###########################################################################

# We need to build the transition matrix P^pi and the reward vector r^pi
# for the policy above, then solve the linear system:
#    (I - gamma * P^pi) * v = r^pi

# Build P^pi: an nS x nS matrix
P_pi = np.zeros((env.nS, env.nS))

# Build r^pi: an nS-dimensional vector
r_pi = np.zeros(env.nS)

def policy_prob(s, a):
    """
    Probability that policy pi takes action a in state s.
    a=0->left, a=1->right
    """
    if s <= 2:
        # prob 0.35 for action=0, 0.65 for action=1
        if a==0:
            return 0.35
        else:
            return 0.65
    else:
        # states 4,5 always right -> action=1
        if a==1:
            return 1.0
        else:
            return 0.0

for s in range(env.nS):
    # r^pi(s) = sum_{a} pi(a|s)* R(s,a)
    r_pi[s] = (policy_prob(s,0)*env.R[s,0]) + (policy_prob(s,1)*env.R[s,1])
    
    # P^pi(s->s') = sum_{a} pi(a|s)* P(s->s'|s,a)
    for s_next in range(env.nS):
        P_pi[s, s_next] = (policy_prob(s,0)*env.P[s,0,s_next]
                           + policy_prob(s,1)*env.P[s,1,s_next])

# Now solve for v: (I - gamma P^pi)*v = r^pi
# -> v = inverse(I - gamma P^pi) * r^pi
# We'll use np.linalg.solve for numerical stability
I = np.eye(env.nS)
A = I - gamma * P_pi
b = r_pi

v_exact = np.linalg.solve(A, b)

print("Exact solution of V^pi(s) using linear system:")
for s in range(env.nS):
    print(f"State {s}: {v_exact[s]:.5f}")
print()
