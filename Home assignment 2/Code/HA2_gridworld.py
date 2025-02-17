
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
			P_pi = np.array([[env.P[s, policy1[s], ss] for ss in range(env.nS)] for s in range(env.nS)])
			R_pi = np.array([env.R[s, policy1[s]] for s in range(env.nS)])
			V_opt = np.linalg.inv(np.eye(env.nS) - gamma * P_pi) @ R_pi
			return niter, policy1, V_opt
		else:
			policy2 = policy0
			policy0 = policy1
			policy1 = np.zeros(env.nS, dtype=int)


import numpy as np

def VI(env, epsilon=1e-6, gamma=0.97, anc=False, V_zero=None):
    """
    Value Iteration with optional Anchored VI, following the pseudocode 
    from the provided slide:
    
    - Initialize V with R_max/(1 - gamma) for all states
    - While ||V_{n+1} - V_n|| >= epsilon*(1-gamma)/(2*gamma), do:
        * V_{n+1}(s) = max_a [ r(s,a) + gamma * sum_{s'} P(s'|s,a) V_n(s') ]
    - Return policy from final V.
    
    Additional: If anc=True, do anchored updates instead of T(V).
                i.e.   V_{n+1} = beta_n * V_zero + (1 - beta_n)* T(V_n)
    
    Parameters
    ----------
    env : 4_Room environment
        Must have env.nS, env.nA, env.P[s,a,s'], env.R[s,a].
    epsilon : float
        The parameter for the stopping condition. 
        (We use the condition: diff < epsilon*(1-gamma)/(2*gamma).)
    gamma : float
        Discount factor in (0,1).
    anc : bool
        If True, run Anchored Value Iteration.
    V_zero : np.array or None
        Anchor vector, shape (env.nS,). If None, uses zero-vector for anchor.
    
    Returns
    -------
    iteration : int
        Number of iterations performed.
    policy : np.array of ints, shape (env.nS,)
        The extracted policy from final V.
    V : np.array of floats, shape (env.nS,)
        The final value function.
    """

    nS, nA = env.nS, env.nA

    if V_zero is None:
		# As in the pseudocode, we pick an initial "upper bound" value
        Rmax = np.max(env.R)  # or a known bound if you prefer
        V = np.ones(nS) * (Rmax / (1.0 - gamma))
    else:
        V = V_zero.copy()
    
    # If we do anchored VI and no anchor is provided, default to zero
    if anc:
        if V_zero is None:
            V_zero = np.zeros(nS)

    iteration = 0
    while True:
        iteration += 1
        
        # compute T(V) at each state
        T_of_V = np.zeros(nS)
        for s in range(nS):
            # Q(s,a) = R(s,a) + gamma sum_s' P[s,a,s'] * V[s']
            Q_sa = np.zeros(nA)
            for a in range(nA):
                Q_sa[a] = env.R[s,a] + gamma*np.dot(env.P[s,a], V)
            T_of_V[s] = np.max(Q_sa)

        if anc:
            # anchored update
            # e.g. beta_n = gamma^{-2n}/ sum_{k=0}^n gamma^{-2k}
            # or whichever anchor formula from the exercise
            r = gamma**(-2)
            numerator = (r ** iteration)
            denom = 0.0
            for k in range(iteration+1):
                denom += (r**k)
            beta = numerator / denom
            
            V_new = beta * V_zero + (1.0 - beta)*T_of_V
        else:
            # standard VI update
            V_new = T_of_V

        diff = np.max(np.abs(V_new - V))
        V = V_new

        # stopping condition from your slide:
        # ||V_{n+1} - V_n|| < eps * (1-gamma) / (2 gamma)
        if diff < epsilon*(1.0 - gamma)/(2.0*gamma):
            break

    # Finally, extract the policy
    policy = np.zeros(nS, dtype=int)
    for s in range(nS):
        Q_sa = np.zeros(nA)
        for a in range(nA):
            Q_sa[a] = env.R[s,a] + gamma*np.dot(env.P[s,a], V)
        policy[s] = np.argmax(Q_sa)

    return iteration, policy, V


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


if __name__ == "__main__":

	###########################################################################
	print("\ni) Solve the grid-world task above using PI")
	###########################################################################

	# Run PI on the environment with gamma = 0.97 and print the result.
	env = Four_Room()
	iterations, pi, V_opt = PI(env, 0.97)
	print("Number of iterations for PI = ", iterations)
	print("Optimal policy from PI =", pi)
	print("Optimal value function V* =", np.round(V_opt, 2))
	print(display_4room_policy(pi))

	###########################################################################
	print("\nii) Implement VI and use it to solve the grid-world task above")
	###########################################################################

	# Run PI on the environment with gamma = 0.97 and epsilon = 1e-6.
	env = Four_Room()
	iterations, optimal_policy, optimal_V = VI(env, epsilon=1e-6, gamma=0.97)
	print("Number of iterations for VI = ", iterations)
	print("Optimal policy from VI = ", optimal_policy)
	print("Optimal value function V* = ", np.round(optimal_V, 2))
	print(display_4room_policy(optimal_policy))

	###########################################################################
	print("\niii) Repeat Part (ii) with gamma = 0.998")
	###########################################################################

	# Run PI on the environment with gamma = 0.97 and epsilon = 1e-6.
	env = Four_Room()
	iterations, optimal_policy, optimal_V = VI(env, epsilon=1e-6, gamma=0.998)
	print("Number of iterations for VI = ", iterations)
	print("Optimal policy from VI = ", optimal_policy)
	print("Optimal value function V* = ", np.round(optimal_V, 2))
	print(display_4room_policy(optimal_policy))

	###########################################################################
	print("\n(iv) Anchored Value Iteration with different initial points")
	###########################################################################

	# Example anchor choices:
	anchor_a = np.zeros(env.nS)                             # (a) V0 = 0
	anchor_b = np.ones(env.nS)                              # (b) V0 = 1
	anchor_c = np.random.rand(env.nS) / (1.0 - 0.97)         # (c) uniform random in [0, 1/(1-gamma)]

	# (a) Anc-VI with anchor 0
	env = Four_Room()
	it_a, pi_a, V_a = VI(env, epsilon=1e-6, gamma=0.97, anc=True, V_zero=anchor_a)
	print(f"Anchored VI with anchor=0 took {it_a} iterations.")
	print("Policy = ", pi_a)
	print("Value  = ", np.round(V_a, 4))

	# (b) Anc-VI with anchor=1
	env = Four_Room()
	it_b, pi_b, V_b = VI(env, epsilon=1e-6, gamma=0.97, anc=True, V_zero=anchor_b)
	print(f"\nAnchored VI with anchor=1 took {it_b} iterations.")
	print("Policy = ", pi_b)
	print("Value  = ", np.round(V_b, 4))

	# (c) Anc-VI with anchor ~ uniform random
	env = Four_Room()
	it_c, pi_c, V_c = VI(env, epsilon=1e-6, gamma=0.97, anc=True, V_zero=anchor_c)
	print(f"\nAnchored VI with random anchor took {it_c} iterations.")
	print("Policy = ", pi_c)
	print("Value  = ", np.round(V_c, 4))

	###########################################################################
	print("\n(v) Compare the convergence speed of VI and Anchored VI (with e.g. anchor=0).")
	###########################################################################

	# (a) VI with anchor 0
	env = Four_Room()
	it_a, pi_a, V_a = VI(env, epsilon=1e-6, gamma=0.97, anc=False, V_zero=anchor_a)
	print(f"VI with anchor=0 took {it_a} iterations.")
	print("Policy = ", pi_a)
	print("Value  = ", np.round(V_a, 4))

	# (b) VI with anchor=1
	env = Four_Room()
	it_b, pi_b, V_b = VI(env, epsilon=1e-6, gamma=0.97, anc=False, V_zero=anchor_b)
	print(f"\nVI with anchor=1 took {it_b} iterations.")
	print("Policy = ", pi_b)
	print("Value  = ", np.round(V_b, 4))

	# (c) VI with anchor ~ uniform random
	env = Four_Room()
	it_c, pi_c, V_c = VI(env, epsilon=1e-6, gamma=0.97, anc=False, V_zero=anchor_c)
	print(f"\nVI with random anchor took {it_c} iterations.")
	print("Policy = ", pi_c)
	print("Value  = ", np.round(V_c, 4))