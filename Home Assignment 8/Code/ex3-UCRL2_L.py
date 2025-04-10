# This code is proposed as a reference solution for various exercises of Home Assignements for the OReL course in 2025.
# This solution is tailored for simplicity of understanding and is in no way optimal, nor the only way to implement the different elements!
import numpy as np
import copy as cp
import pylab as pl



####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################

#																	ENVIRONMENTS

####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################




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
	










####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################

#																	VI and PI

####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################






# An implementation of the Value Iteration algorithm for a given environment 'env' in an average reward setting.
# An arbitrary 'max_iter' is a maximum number of iteration, usefull to catch any error in your code!
# Return the number of iterations, the final value, the optimal policy and the gain.
def VI(env, max_iter = 10**3, epsilon = 10**(-2)):

	# The variable containing the optimal policy estimate at the current iteration.
	policy = np.zeros(env.nS, dtype=int)
	niter = 0

	# Initialise the value and epsilon as proposed in the course.
	V0 = np.zeros(env.nS)
	V1 = np.zeros(env.nS)

	# The main loop of the Value Iteration algorithm.
	while True:
		niter += 1
		for s in range(env.nS):
			for a in range(env.nA):
				temp = env.R[s, a] + sum([V * p for (V, p) in zip(V0, env.P[s, a])])
				if (a == 0) or (temp > V1[s]):
					V1[s] = temp
					policy[s] = a
		
		# Testing the stopping criterion (+1 abitrary stop when 'max_iter' is reached).
		gain = 0.5*(max(V1 - V0) + min(V1 - V0))
		diff  = [abs(x - y) for (x, y) in zip(V1, V0)]
		if (max(diff) - min(diff)) < epsilon:
			return niter, V0, policy, gain
		else:
			V0 = V1
			V1 = np.zeros(env.nS)
		if niter > max_iter:
			print("No convergence in VI after: ", max_iter, " steps!")
			return niter, V0, policy, gain











####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################

#																	UCRL2

####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################







# A simple implementation of the UCRL2 algorithm from Jacksh et al. 2010 with improved L1-Laplace confidence intervals.
class UCRL2_L:
	def __init__(self, nS, nA, gamma, epsilon = 0.01, delta = 0.05):
		self.nS = nS
		self.nA = nA
		self.gamma = gamma
		self.delta = delta
		self.epsilon = epsilon
		self.s = None

		# The "counter" variables:
		self.Nk = np.zeros((self.nS, self.nA), dtype=int) # Number of occurences of (s, a) at the end of the last episode.
		self.Nsas = np.zeros((self.nS, self.nA, self.nS), dtype=int) # Number of occureces of (s, a, s').
		self.Rsa = np.zeros((self.nS, self.nA)) # Cumulated reward observed for (s, a).
		self.vk = np.zeros((self.nS, self.nA)) # Number of occurences of (s, a) in the current episode.

		# The "estimates" variables:
		self.hatP = np.zeros((self.nS, self.nA, self.nS)) # Estimate of the transition matrix.
		self.hatR = np.zeros((self.nS, self.nA))
		
		# Confidence intervals:
		self.confR = np.zeros((self.nS, self.nA))
		self.confP = np.zeros((self.nS, self.nA))

		# The current policy (updated at each episode).
		self.policy = np.zeros((self.nS,), dtype=int)

		self.ep_count = 0  # Initialize episode counter

	# Auxiliary function to update N the current state-action count.
	def updateN(self):
		for s in range(self.nS):
			for a in range(self.nA):
				self.Nk[s, a] += self.vk[s, a]

	# Update the confidence intervals. Set with Laplace-L1 confidence intervals!
	def confidence(self):
		d = self.delta / (self.nS * self.nA)
		for s in range(self.nS):
			for a in range(self.nA):
				n = max(1, self.Nk[s, a])
				self.confR[s, a] = np.sqrt(((1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / (2 * n))
				self.confP[s, a] = np.sqrt((2 * (1 + 1 / n) * np.log(np.sqrt(n + 1) * (2**(self.nS) - 2) / d)) / n)
	
	# Computing the maximum proba in the Extended Value Iteration for given state s and action a.
	# From UCRL2 jacksh et al. 2010.
	def max_proba(self, sorted_indices, s, a):
		min1 = min([1, self.hatP[s, a, sorted_indices[-1]] + (self.confP[s, a] / 2)])
		max_p = np.zeros(self.nS)
		if min1 == 1:
			max_p[sorted_indices[-1]] = 1
		else:
			max_p = cp.deepcopy(self.hatP[s, a])
			max_p[sorted_indices[-1]] += self.confP[s, a] / 2
			l = 0
			while sum(max_p) > 1:
				max_p[sorted_indices[l]] = max([0, 1 - sum(max_p) + max_p[sorted_indices[l]]])
				l += 1
		return max_p


	# The Extended Value Iteration, perform an optimisitc VI over a set of MDP.
	def EVI(self, max_iter = 2*10**2, epsilon = 10**(-2)):
		niter = 0
		sorted_indices = np.arange(self.nS)
		action_noise = [(np.random.random_sample() * 0.1 * min((1e-6, epsilon))) for _ in range(self.nA)]

		# The variable containing the optimistic policy estimate at the current iteration.
		policy = np.zeros(self.nS, dtype=int)

		# Initialise the value and epsilon as proposed in the course.
		V0 = np.zeros(self.nS)# NB: setting it to the bias obtained at the last episode can help speeding up the convergence significantly!
		V1 = np.zeros(self.nS)

		# The main loop of the Value Iteration algorithm.
		while True:
			niter += 1
			for s in range(self.nS):
				for a in range(self.nA):
					maxp = self.max_proba(sorted_indices, s, a)
					temp = min(1, self.hatR[s, a] + self.confR[s, a]) + sum([V * p for (V, p) in zip(V0, maxp)])
					if (a == 0) or ((temp + action_noise[a]) > (V1[s] + action_noise[self.policy[s]])): # Using a noise to randomize the choice when equals.
						V1[s] = temp
						policy[s] = a

			# Testing the stopping criterion (+1 abitrary stop when 'max_iter' is reached).
			diff  = [abs(x - y) for (x, y) in zip(V1, V0)]
			if (max(diff) - min(diff)) < epsilon:
				return policy
			else:
				V0 = V1
				V1 = np.zeros(self.nS)
				sorted_indices = np.argsort(V0)
			if niter > max_iter:
				print("No convergence in EVI after: ", max_iter, " steps!")
				return policy

	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):

		self.ep_count += 1  # Increment episode counter at the start of a new episode

		self.updateN() # We update the counter Nk.
		self.vk = np.zeros((self.nS, self.nA))

		# Update estimates, note that the estimates are 0 at first, the optimistic strategy making that irrelevant.
		for s in range(self.nS):
			for a in range(self.nA):
				div = max([1, self.Nk[s, a]])
				self.hatR[s, a] = self.Rsa[s, a] / div
				for next_s in range(self.nS):
					self.hatP[s, a, next_s] = self.Nsas[s, a, next_s] / div

		# Update the confidence intervals and policy.
		self.confidence()
		self.policy = self.EVI()

	# To reinitialize the model and a give the new initial state init.
	def reset(self, init):
		# The "counter" variables:
		self.Nk = np.zeros((self.nS, self.nA), dtype=int) # Number of occurences of (s, a) at the end of the last episode.
		self.Nsas = np.zeros((self.nS, self.nA, self.nS), dtype=int) # Number of occureces of (s, a, s').
		self.Rsa = np.zeros((self.nS, self.nA)) # Cumulated reward observed for (s, a).
		self.vk = np.zeros((self.nS, self.nA)) # Number of occurences of (s, a) in the current episode.

		# The "estimates" variables:
		self.hatP = np.zeros((self.nS, self.nA, self.nS)) # Estimate of the transition matrix.
		self.hatR = np.zeros((self.nS, self.nA))
		
		# Confidence intervals:
		self.confR = np.zeros((self.nS, self.nA))
		self.confP = np.zeros((self.nS, self.nA))

		# The current policy (updated at each episode).
		self.policy = np.zeros((self.nS,), dtype=int)

		# Set the initial state and last action:
		self.s = init
		self.last_action = -1

		# Start the first episode.
		self.new_episode()

	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self,state, reward):
		if self.last_action >= 0: # Update if not first action.
			self.Nsas[self.s, self.last_action, state] += 1
			self.Rsa[self.s, self.last_action] += reward
		
		action = self.policy[state]
		if self.vk[state, action] > max([1, self.Nk[state, action]]): # Stoppping criterion
			self.new_episode()
			action  = self.policy[state]
		
		# Update the variables:
		self.vk[state, action] += 1
		self.s = state
		self.last_action = action

		return action, self.policy










####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################

#																	Running experiments

####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################








# Plotting function.
def plot(data, names, y_label="Regret", exp_name="cumulativeRegret", horizontal_line=None):
    # Determine the time horizon based on the first experiment's data
    timeHorizon = len(data[0][0])
    colors = ['black', 'blue', 'purple', 'cyan', 'yellow', 'orange', 'red']
    nbFigure = pl.gcf().number + 1

    # Calculate average data for each experiment and plot it
    avg_data = []
    pl.figure(nbFigure)
    for i in range(len(data)):
        avg_data.append(np.mean(data[i], axis=0))
        pl.plot(avg_data[i], label=names[i], color=colors[i % len(colors)])

    # Compute standard deviation and add error bars
    step = timeHorizon // 10
    for i in range(len(data)):
        std_data = 1.96 * np.std(data[i], axis=0) / np.sqrt(len(data[i]))
        pl.errorbar(np.arange(0, timeHorizon, step),
                    avg_data[i][0:timeHorizon:step],
                    std_data[0:timeHorizon:step],
                    color=colors[i % len(colors)], linestyle='None', capsize=10)
    
    # If a horizontal line value is provided, draw it
    if horizontal_line is not None:
        pl.axhline(y=horizontal_line, color='red', linestyle='--', label='Optimal Gain')
    
    # Label and format the plot
    pl.legend()
    pl.xlabel("Time steps", fontsize=13, fontname="Arial")
    pl.ylabel(y_label, fontsize=13, fontname="Arial")
    pl.ticklabel_format(axis='both', useMathText=True, useOffset=True, style='sci', scilimits=(0, 0))
    
    # Save the plot using a constructed file name
    file_name = ""
    for n in names:
        file_name += n + "_"
    pl.savefig("Home Assignment 8/Code/3/Figure_" + file_name + exp_name)

# Test function, plotting the cumulative regret.
def run():
	# Set the environment:
	nS = 6
	env = riverswim(nS)
	epsilon = 0.01
	delta = 0.05

	# Set the time horizon:
	T = 2*10**4
	nb_Replicates = 100

	# Set the learning agents:
	UCRL2L = UCRL2_L(nS, 2, epsilon, delta)

	# Set the variables used for plotting.
	cumregret_UCRL2L = [[0] for _ in range(nb_Replicates)]

	# Estimate the optimal gain.
	print("Estimating the optimal gain...",)
	_, _, _, gstar = VI(env, 10**6, 10**(-6))

	# Run the experiments:
	print("Running experiments...")
	for i in range(nb_Replicates):
		# Running an instance of UCRL2-L:
		env.reset()
		UCRL2L.reset(env.s)
		reward = 0
		new_s = env.s
		for t in range(T):
			action, _ = UCRL2L.play(new_s, reward)
			new_s, reward = env.step(action)
			cumregret_UCRL2L[i].append(cumregret_UCRL2L[i][-1]+ gstar - reward)
		print("|" + "#"*int(i/((nb_Replicates - 1) / 33)) + " "*(33 - int(i/((nb_Replicates - 1) / 33))) + "|", end="\r") # Making a rudimentary progress bar!
	
	# Plot and finish.
	print("\nPlotting...")
	plot([cumregret_UCRL2L], ["UCRL2_L"], y_label = "Cumulative Regret", exp_name = "cumulative_regret")
	print('Done!')


run()

# New class: UCRL2 (original confidence sets per Jaksch et al. 2010)
class UCRL2:
    def __init__(self, nS, nA, gamma, epsilon=0.01, delta=0.05):
        self.nS = nS
        self.nA = nA
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.s = None
        # Counters: number of visits and counts for (s,a,s')
        self.Nk = np.zeros((nS, nA), dtype=int)
        self.Nsas = np.zeros((nS, nA, nS), dtype=int)
        self.Rsa = np.zeros((nS, nA))
        self.vk = np.zeros((nS, nA))
        # Estimates for the dynamics and rewards
        self.hatP = np.zeros((nS, nA, nS))
        self.hatR = np.zeros((nS, nA))
        # Confidence intervals for rewards and transitions
        self.confR = np.zeros((nS, nA))
        self.confP = np.zeros((nS, nA))
        # Optimistic policy over states
        self.policy = np.zeros(nS, dtype=int)
        # Counter for episodes
        self.ep_count = 0

    def updateN(self):
        for s in range(self.nS):
            for a in range(self.nA):
                self.Nk[s, a] += self.vk[s, a]

    # Confidence intervals per Jaksch et al. (2010):
    # |Rhat - R'| <= sqrt((3.5*log((2*S*A*n)/delta))/n)
    # ||Phat - P'||₁ <= sqrt((14*S*log((2*A*n)/delta))/n)
    def confidence(self):
        for s in range(self.nS):
            for a in range(self.nA):
                n = max(1, self.Nk[s, a])
                self.confR[s, a] = np.sqrt((3.5 * np.log((2 * self.nS * self.nA * n) / self.delta)) / n)
                self.confP[s, a] = np.sqrt((14 * self.nS * np.log((2 * self.nA * n) / self.delta)) / n)

    def max_proba(self, sorted_indices, s, a):
        # Optimistically choose a transition model within the UCRL2 confidence set
        min_val = min(1, self.hatP[s, a, sorted_indices[-1]] + (self.confP[s, a] / 2))
        max_p = np.zeros(self.nS)
        if min_val == 1:
            max_p[sorted_indices[-1]] = 1
        else:
            max_p = cp.deepcopy(self.hatP[s, a])
            max_p[sorted_indices[-1]] += self.confP[s, a] / 2
            l = 0
            while sum(max_p) > 1:
                max_p[sorted_indices[l]] = max(0, 1 - sum(max_p) + max_p[sorted_indices[l]])
                l += 1
        return max_p

    def EVI(self, max_iter=200, epsilon=1e-2):
        niter = 0
        sorted_indices = np.arange(self.nS)
        # Small random noise to break ties
        action_noise = [np.random.random_sample() * 0.1 * min(1e-6, epsilon) for _ in range(self.nA)]
        policy = np.zeros(self.nS, dtype=int)
        V0 = np.zeros(self.nS)
        V1 = np.zeros(self.nS)
        while True:
            niter += 1
            for s in range(self.nS):
                for a in range(self.nA):
                    maxp = self.max_proba(sorted_indices, s, a)
                    temp = min(1, self.hatR[s, a] + self.confR[s, a]) + np.dot(V0, maxp)
                    if (a == 0) or ((temp + action_noise[a]) > (V1[s] + action_noise[policy[s]])):
                        V1[s] = temp
                        policy[s] = a
            diff = [abs(x - y) for (x, y) in zip(V1, V0)]
            if (max(diff) - min(diff)) < epsilon:
                return policy
            else:
                V0 = V1
                V1 = np.zeros(self.nS)
                sorted_indices = np.argsort(V0)
            if niter > max_iter:
                print("No convergence in EVI after", max_iter, "steps!")
                return policy

    def new_episode(self):
        self.ep_count += 1
        self.updateN()
        self.vk = np.zeros((self.nS, self.nA))
        # Update empirical estimates based on counts so far
        for s in range(self.nS):
            for a in range(self.nA):
                div = max(1, self.Nk[s, a])
                self.hatR[s, a] = self.Rsa[s, a] / div
                for next_s in range(self.nS):
                    self.hatP[s, a, next_s] = self.Nsas[s, a, next_s] / div
        self.confidence()
        self.policy = self.EVI()

    def reset(self, init):
        self.Nk = np.zeros((self.nS, self.nA), dtype=int)
        self.Nsas = np.zeros((self.nS, self.nA, self.nS), dtype=int)
        self.Rsa = np.zeros((self.nS, self.nA))
        self.vk = np.zeros((self.nS, self.nA))
        self.hatP = np.zeros((self.nS, self.nA, self.nS))
        self.hatR = np.zeros((self.nS, self.nA))
        self.confR = np.zeros((self.nS, self.nA))
        self.confP = np.zeros((self.nS, self.nA))
        self.policy = np.zeros(self.nS, dtype=int)
        self.s = init
        self.last_action = -1
        self.ep_count = 0
        self.new_episode()

    def play(self, state, reward):
        if self.last_action >= 0:  # Not first move: update counts based on previous (s, a)
            self.Nsas[self.s, self.last_action, state] += 1
            self.Rsa[self.s, self.last_action] += reward
        action = self.policy[state]
        if self.vk[state, action] > max(1, self.Nk[state, action]):
            self.new_episode()
            action = self.policy[state]
        self.vk[state, action] += 1
        self.s = state
        self.last_action = action
        return action, self.policy


# Modified experiment: run both UCRL2 and UCRL2-L on 6-state RiverSwim
def run_experiment():
    nS = 6
    T = int(3.5e5)
    nb_Replicates = 50
    epsilon = 0.01

    # Estimate the optimal gain g* via Value Iteration
    env_for_vi = riverswim(nS)
    _, _, _, gstar = VI(env_for_vi, 10**6, 1e-6)

    # Initialize containers for regrets, gain curves, and episode counts
    regret_ucrl2 = []     # For UCRL2
    regret_ucrl2_l = []   # For UCRL2-L
    gain_ucrl2 = []
    gain_ucrl2_l = []
    episodes_ucrl2 = []
    episodes_ucrl2_l = []

    for rep in range(nb_Replicates):
        # ------------------------
        # UCRL2 experiment (δ = 0.05)
        env1 = riverswim(nS)
        env1.reset()
        agent_ucrl2 = UCRL2(nS, 2, epsilon, delta=0.05)
        agent_ucrl2.reset(env1.s)
        cum_regret = [0]
        rewards = []
        current_state = env1.s
        # For UCRL2, initialize reward (the first reward is taken as 0)
        reward = 0
        for t in range(T):
            action, _ = agent_ucrl2.play(current_state, reward)
            next_state, reward = env1.step(action)
            cum_regret.append(cum_regret[-1] + gstar - reward)
            rewards.append(reward)
            current_state = next_state
        regret_ucrl2.append(cum_regret)
        gain_ucrl2.append(np.cumsum(rewards) / np.arange(1, T + 1))
        episodes_ucrl2.append(agent_ucrl2.ep_count)

        # ------------------------
        # UCRL2-L experiment (δ = 0.0125)
        env2 = riverswim(nS)
        env2.reset()
        agent_ucrl2_l = UCRL2_L(nS, 2, epsilon, delta=0.0125)
        agent_ucrl2_l.reset(env2.s)
        cum_regret_l = [0]
        rewards_l = []
        current_state_l = env2.s
        reward_l = 0
        for t in range(T):
            action, _ = agent_ucrl2_l.play(current_state_l, reward_l)
            next_state_l, reward_l = env2.step(action)
            cum_regret_l.append(cum_regret_l[-1] + gstar - reward_l)
            rewards_l.append(reward_l)
            current_state_l = next_state_l
        regret_ucrl2_l.append(cum_regret_l)
        gain_ucrl2_l.append(np.cumsum(rewards_l) / np.arange(1, T + 1))
        episodes_ucrl2_l.append(agent_ucrl2_l.ep_count)

        print(f"Replicate {rep + 1}/{nb_Replicates} completed.")

    # Plot cumulative regret (using the provided 'plot' function)
    plot([regret_ucrl2, regret_ucrl2_l],
         ["UCRL2", "UCRL2_L"],
         y_label="Cumulative Regret",
         exp_name="cumulative_regret_comparison")
    
    # Plot average gain curves, adding a horizontal line for the optimal gain (gstar)
    plot([gain_ucrl2, gain_ucrl2_l],
		["UCRL2", "UCRL2_L"],
		y_label="Empirical Average Gain",
		exp_name="gain_comparison",
		horizontal_line=gstar)

    # Report average number of episodes
    print("Average number of episodes initiated:")
    print("UCRL2:", np.mean(episodes_ucrl2))
    print("UCRL2_L:", np.mean(episodes_ucrl2_l))


# Run the experiment
run_experiment()
