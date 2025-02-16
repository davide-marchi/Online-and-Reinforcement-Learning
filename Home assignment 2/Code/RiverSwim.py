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
	





# A example of application: running the policy "go to the right" for 10 steps and print the sequence of states.
env = riverswim(6)
l = [0] # Initial state arbitrarily set at 0 in the class above.
for _ in range(10):
	l.append(env.step(1)[0])
print("Sequence of states = ", l)