
# Written by Hippolyte Bourel.
# This code is proposed as a reference solution for various exercises of Home Assignements for the OReL course in 2024.
# This solution is tailored for simplicity of understanding and is in no way optimal, nor the only way to implement the different elements!
import numpy as np


# A simple 4-room gridworld implementation with a grid of 7x7 for a total of 20 states (the walls do not count!).
# We arbitrarily chose the actions '0' = 'go up', '1' = 'go right', '2'  = 'go down' thus '3' = 'go left'
# Finally the state '0' is the top-left corner, 'nS - 1' is the down-right corner.
# The agent is unable to leave the state '19' (down-right corner) and receive a reward of 1 for all actions in this state.
class Four_Room():

	def __init__(self):
		self.nS = 20
		nS = self.nS
		self.nA = 4

		self.map = [[-1, -1, -1, -1, -1, -1, -1],
					[-1,  0,  1,  2,  3,  4, -1],
					[-1,  5,  6, -1,  7,  8, -1],
					[-1,  9, -1, -1, 10, -1, -1],
					[-1, 11, 12, 13, 14, 15, -1],
					[-1, 16, 17, -1, 18, 19, -1],
					[-1, -1, -1, -1, -1, -1, -1]]
		map = np.array(self.map)

		# We build the transitions matrix P using the map.
		self.P = np.zeros((nS, 4, nS))

		for s in range(nS):
			temp = np.where(s == map)
			y, x = temp[0][0], temp[1][0]
			up = map[x, y-1]
			right = map[x+1, y]
			down = map[x, y+1]
			left = map[x-1, y]

			# Action 0: go up.
			a = 0
			self.P[s, a, s] += 0.1
			# Up
			if up == -1:
				self.P[s, a, s] += 0.7
			else:
				self.P[s, a, up] += 0.7
			# Right
			if right == -1:
				self.P[s, a, s] += 0.1
			else:
				self.P[s, a, right] += 0.1
			# Left
			if left == -1:
				self.P[s, a, s] += 0.1
			else:
				self.P[s, a, left] += 0.1
			
			# Action 1: go right.
			a = 1
			self.P[s, a, s] += 0.1
			# Up
			if up == -1:
				self.P[s, a, s] += 0.1
			else:
				self.P[s, a, up] += 0.1
			# Right
			if right == -1:
				self.P[s, a, s] += 0.7
			else:
				self.P[s, a, right] += 0.7
			# Down
			if down == -1:
				self.P[s, a, s] += 0.1
			else:
				self.P[s, a, down] += 0.1
			
			# Action 2: go down.
			a = 2
			self.P[s, a, s] += 0.1
			# Right
			if right == -1:
				self.P[s, a, s] += 0.1
			else:
				self.P[s, a, right] += 0.1
			# Down
			if down == -1:
				self.P[s, a, s] += 0.7
			else:
				self.P[s, a, down] += 0.7
			# Left
			if left == -1:
				self.P[s, a, s] += 0.1
			else:
				self.P[s, a, left] += 0.1

			# Action 3: go left.
			a = 3
			self.P[s, a, s] += 0.1
			# Up
			if up == -1:
				self.P[s, a, s] += 0.1
			else:
				self.P[s, a, up] += 0.1
			# Down
			if down == -1:
				self.P[s, a, s] += 0.1
			else:
				self.P[s, a, down] += 0.1
			# Left
			if left == -1:
				self.P[s, a, s] += 0.7
			else:
				self.P[s, a, left] += 0.7
			
			# Set to teleport back when in the rewarding state.
			if s == self.nS - 1:
				for a in range(4):
					for ss in range(self.nS):
						self.P[s, a, ss] = 0
						if ss == s:
							self.P[s, a, ss] = 1

			
		# We build the reward matrix R.
		self.R = np.zeros((nS, 4))
		for a in range(4):
			self.R[nS - 1, a] = 1

		# We (arbitrarily) set the initial state in the top-left corner.
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







# An implementation of the PI algorithm, using a matrix inversion to do the policy evaluation step.
# Return the number of iterations and the policy.
def PI(env, gamma = 0.9):

	# Initialisation of the variables.
	policy0 = np.random.randint(env.nA, size = env.nS)
	policy1 = np.zeros(env.nS, dtype = int)
	niter = 0

	# The main loop of the PI algorithm.
	while True:
		niter += 1

		# Policy evaluation step.
		P_pi = np.array([[env.P[s, policy0[s], ss] for ss in range(env.nS)] for s in range(env.nS)])
		R_pi = np.array([env.R[s, policy0[s]] for s in range(env.nS)])
		V0 = np.linalg.inv((np.eye(env.nS) - gamma * P_pi)) @ R_pi
		V1 = np.zeros(env.nS)

		# Updating the policy.
		for s in range(env.nS):
			for a in range(env.nA):
				temp = env.R[s, a] + gamma * sum([u * p for (u, p) in zip(V0, env.P[s, a])])
				if (a == 0) or (temp > V1[s]):
					V1[s] = temp
					policy1[s] = a

		# Testing if the policy changed or not.
		test = True
		for s in range(env.nS):
			if policy0[s] != policy1[s]:
				test = False
				break
		
		Vdiff = [V1[i] - V0[i] for i in range(env.nS)]

		# If the policy did not change or the change was due to machine limitation in numerical values return the result.	
		if test or (max(Vdiff) < 10**(-12)):
			return niter, policy1
		else:
			policy2 = policy0
			policy0 = policy1
			policy1 = np.zeros(env.nS, dtype=int)



# A naive function to output a readble matrix from a policy on the 4-room environment.
def display_4room_policy(policy):
	map = np.array([[-1, -1, -1, -1, -1, -1, -1],
					[-1,  0,  1,  2,  3,  4, -1],
					[-1,  5,  6, -1,  7,  8, -1],
					[-1,  9, -1, -1, 10, -1, -1],
					[-1, 11, 12, 13, 14, 15, -1],
					[-1, 16, 17, -1, 18, 19, -1],
					[-1, -1, -1, -1, -1, -1, -1]])
	
	res = []

	for i in range(7):
		temp = []
		for j in range(7):
			if map[i][j] == -1:
				temp.append("Wall ")
			elif policy[map[i][j]] == 0:
				temp.append(" Up  ")
			elif policy[map[i][j]] == 1:
				temp.append("Right")
			elif policy[map[i][j]] == 2:
				temp.append("Down ")
			elif policy[map[i][j]] == 3:
				temp.append("Left ")
		
		res.append(temp)

	return np.array(res)

	



# Run PI on the environment with gamma = 0.95 and print the result.
env = Four_Room()
_, pi = PI(env, 0.95)
print(display_4room_policy(pi))