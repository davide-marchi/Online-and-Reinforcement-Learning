
# Written by Hippolyte Bourel.
# This code is proposed as a reference solution for various exercises of Home Assignements for the OReL course in 2024.
# This solution is tailored for simplicity of understanding and is in no way optimal, nor the only way to implement the different elements!
import numpy as np







# A simple 4-room gridworld implementation with a grid of 7x7 for a total of 20 states (the walls do not count!).
# We arbitrarily chose the actions '0' = 'go up', '1' = 'go right', '2'  = 'go down' thus '3' = 'go left'
# Finally the state '0' is the top-left corner, 'nS - 1' is the down-right corner.
# The agent is teleported back to the the initial state '0' (top-left corner) ,  whenever performing any action in rewarding state '19' (down-right corner).
class Four_Room_Teleportation():

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
			
			# Set to teleport back to top-left corner when in the rewarding state.
			if s == self.nS - 1:
				for a in range(4):
					for ss in range(self.nS):
						self.P[s, a, ss] = 0
						if ss == 0:
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

	

def average_reward_vi(mdp, epsilon=1e-6, max_iter=10000):
    """
    Value Iteration for Average-Reward MDPs.
    
    Parameters:
        mdp      : The MDP instance with attributes:
                   - nS: number of states
                   - nA: number of actions
                   - R : reward matrix (shape: [nS, nA])
                   - P : transition probability matrix (shape: [nS, nA, nS])
        epsilon  : Convergence threshold (using the span of the value differences)
        max_iter : Maximum number of iterations
    
    Returns:
        policy : Optimal policy as a numpy array (one action per state)
        gain   : Estimated optimal gain (g*)
        bias   : Estimated bias function (b*), normalized so that min(b*) = 0
    """
    # Initialize the value function arbitrarily (here, zeros)
    V = np.zeros(mdp.nS)
    
    for iteration in range(max_iter):
        V_next = np.zeros(mdp.nS)
        # Update V_next for each state s using the Bellman operator:
        # V_next(s) = max_a [ R(s, a) + sum_x P(x|s,a) * V(x) ]
        for s in range(mdp.nS):
            action_values = []
            for a in range(mdp.nA):
                q_sa = mdp.R[s, a] + np.dot(mdp.P[s, a], V)
                action_values.append(q_sa)
            V_next[s] = max(action_values)
        
        # Compute the difference (increment) vector
        diff = V_next - V
        # The span (max difference minus min difference) is our stopping criterion
        span_diff = np.max(diff) - np.min(diff)
        V = V_next.copy()
        
        if span_diff < epsilon:
            print(f"Converged after {iteration+1} iterations with span {span_diff:.2e}.")
            break
    else:
        print("Warning: Maximum iterations reached without full convergence.")
    
    # Estimate the optimal gain g* as the average of the maximum and minimum differences.
    gain = 0.5 * (np.max(diff) + np.min(diff))
    # The bias function is approximated by normalizing V (bias is defined up to an additive constant)
    bias = V - np.min(V)
    
    # Derive the optimal policy: for each state, choose the action maximizing:
    # Q(s, a) = R(s, a) + sum_x P(x|s,a) * V(x)
    policy = np.zeros(mdp.nS, dtype=int)
    for s in range(mdp.nS):
        best_val = -np.inf
        best_a = 0
        for a in range(mdp.nA):
            q_sa = mdp.R[s, a] + np.dot(mdp.P[s, a], V)
            if q_sa > best_val:
                best_val = q_sa
                best_a = a
        policy[s] = best_a
    
    return policy, gain, bias

if __name__ == '__main__':
    # Create an instance of the 4-room grid-world environment.
    env = Four_Room_Teleportation()
    
    # Set the convergence threshold (epsilon) as suggested (1e-6)
    epsilon = 1e-6
    
    # Run Value Iteration for the average-reward formulation.
    policy, gain, bias = average_reward_vi(env, epsilon=epsilon)
    
    # Output the results.
    print("Optimal Gain (g*):", gain)
    span_bias = np.max(bias) - np.min(bias)
    print("Span of the Optimal Bias (sp(b*)):", span_bias)
    
    # Visualize the optimal policy using the provided grid formatting function.
    policy_grid = display_4room_policy(policy)
    print("Optimal Policy Grid:")
    print(policy_grid)