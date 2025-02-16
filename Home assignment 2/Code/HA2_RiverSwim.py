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
# i) Monte Carlo approximation of V^pi
###########################################################################

# Parameters for the simulation
gamma = 0.96         # discount factor - from the exercise requirements
num_episodes = 50    # number of episodes to run (more = better approximation but slower)
T = 300             # max steps per episode (should be enough to converge)

# Policy implementation:
# For the given policy:
# - If we're in states 0,1,2: mostly go right (65% chance) but sometimes left (35%)
# - If we're in states 3,4: always go right (trying to get that sweet reward at the end)
def sample_action_from_policy(s):
    """
    Gets an action based on our policy.
    Note to self: 0=left, 1=right!
    """
    if s <= 2:
        # Using np.random.choice to implement the 65-35 split
        return np.random.choice([0,1], p=[0.35,0.65])
    else:
        # In states 3,4 we always go right
        return 1

# Initialize our river environment with 5 states (0-4 not 1-5, but we can add 1 later)
env = riverswim(5)

# Array to store our MC estimates - one value for each state
Vhat_MC = np.zeros(env.nS)

# Main Monte Carlo estimation loop
# Need to try from every starting state to get full V function
for s in range(env.nS):
    returns_sum = 0.0    # Accumulator for returns from this state

    # Run multiple episodes to get a good average
    for _ in range(num_episodes):
        env.s = s        # Force starting state
        
        # Variables for calculating discounted return
        G = 0.0          # Total return for this episode
        discount = 1.0   # Current discount factor (will multiply by gamma each step)
        
        # Run one episode
        for t in range(T):
            a = sample_action_from_policy(env.s)
            s_next, r = env.step(a)
            
            G += discount * r    # Add discounted reward to return
            discount *= gamma    # Update discount for next step
            # State automatically updates in env.step

        returns_sum += G

    # Average returns to get value estimate for this state
    Vhat_MC[s] = returns_sum / num_episodes

print("Monte Carlo estimates of V^pi(s) for s=0..4:")
for s in range(env.nS):
    print(f"State {s+1}: {Vhat_MC[s]:.5f}")
print()

###########################################################################
# ii) Exact value of V^pi by direct computation
###########################################################################

# Need to solve (I - gamma*P^pi)v = r^pi
# First build P^pi and r^pi for our policy
P_pi = np.zeros((env.nS, env.nS))
r_pi = np.zeros(env.nS)

def policy_prob(s, a):
    """
    Helper function to get pi(a|s)
    Basically translates our policy into probabilities
    """
    if s <= 2:
        # States 0,1,2: 35% left, 65% right
        return 0.35 if a==0 else 0.65
    else:
        # States 3,4: always right
        return 0.0 if a==0 else 1.0

# Build P^pi and r^pi matrices based on our policy
for s in range(env.nS):
    # r^pi is weighted average of rewards for each action
    r_pi[s] = (policy_prob(s,0)*env.R[s,0]) + (policy_prob(s,1)*env.R[s,1])
    
    # P^pi combines transition probs weighted by policy probs
    for s_next in range(env.nS):
        P_pi[s, s_next] = (policy_prob(s,0)*env.P[s,0,s_next]
                           + policy_prob(s,1)*env.P[s,1,s_next])

# Solve the system (I - gamma*P^pi)v = r^pi
# Using numpy's solver because it's more stable than matrix inversion
I = np.eye(env.nS)
A = I - gamma * P_pi
b = r_pi

v_exact = np.linalg.solve(A, b)

print("Exact solution of V^pi(s) using linear system:")
for s in range(env.nS):
    print(f"State {s+1}: {v_exact[s]:.5f}")
print()
