import numpy as np
import math

import matplotlib.pyplot as plt

import random
from random import choice
from random import choices

from .magic import Magic

class SimulatedAnnealing:
	
	def __init__(self, magic, n = 5, iterations = 100, debug = False):
		self.n = n
		self.iterations = iterations
		self.magic = Magic(self.n)
		self.magic.set_magic(magic)

		self.debug = debug

		self.path = []


	def get_random_descendant(self, curr_state):
		''' Randomly picks one of the randomly generated 
			descendats.

		Arguments:
			curr_state (Magic obj.): instance representing
									 current state
		Returns:
			(list): [0] -- randomly chosen descendant of the
						   current state
					[1] -- objective fcn value of the current
						   state
		'''
		descendants = curr_state.generate_descendants(500)
		random_desc = (choice(descendants))

		return [random_desc, random_desc.objective_fcn()]


	def get_solution(self):
		''' Returns the magic square using simulated
			annealing algorithm. The temperature schedule
			is achieved by multiplying the initial temperature
			with the cooling constant.

		Returns:
			(list): [0] -- Magic square selected after specified
						   number of iterations
					[1] -- Number of constraints selected magic
						   square breaks.
					[2] -- Achieved number of iterations
		'''
		T0 = 50	# initial temperature
		t = T0	# current temperature
		cooling = 0.9988	# temperature cooling coefficient

		P_vec = []	# list containing probability of picking the worse solution
		viol = []	# list containing the number of violations at each iter.
		T = []		# list representing temperature over time

		tt = []		# list containg temperatures at iterations where we consider
					# worse solution

		curr_state = self.magic	# current state/magic square
		curr_val = curr_state.objective_fcn()	# current state object fcn value
		viol.append(curr_state.number_of_violations())

		for k in range(0, self.iterations):
			# Updating the current temperature
			t *= cooling
			if t==0:
				print(k+1)
			T.append(t)
			[next_state, next_val] = self.get_random_descendant(curr_state)
			viol.append(next_state.number_of_violations())

			if next_state.number_of_violations() == 0:	# check if current state is goal state
				if self.debug:
					fig, ax = plt.subplots(nrows = 3, ncols = 1, figsize=(10,5))

					ax[0].plot(T)
					ax[0].set_title("Temperature", fontsize = 9)

					ax[1].plot(tt, P_vec)
					ax[1].set_title("Probability", fontsize = 9)

					ax[2].plot(viol)
					ax[2].set_title("Number of violations", fontsize = 9)

					plt.show()

				return [next_state, 0, k+1]

			if next_val>=curr_val:
				curr_state = next_state
				curr_val = next_val
			else:
				tt.append(k)
				delta_E = curr_val - next_val

				try:
					P = np.exp((-delta_E)/(t))
				except:
					P = 0

				if math.isnan(P):
					P = 0

				P_vec.append(P)	# probability of taking the worse solution

				P_take = np.random.sample()

				if P_take<P:
					curr_state = next_state
					curr_val = next_val

			if((k+1)%100==0) and self.debug:
				print("Iteration: {}, BEST: {}".format(k+1, curr_state.number_of_violations()))


		if self.debug:
			fig, ax = plt.subplots(nrows = 3, ncols = 1, figsize=(10,5))

			ax[0].plot(T)
			ax[0].set_title("Temperature", fontsize = 9)

			ax[1].plot(tt, P_vec)
			ax[1].set_title("Probability", fontsize = 9)

			ax[2].plot(viol)
			ax[2].set_title("Number of violations", fontsize = 9)

			plt.show()

		return (curr_state, curr_state.number_of_violations(), self.iterations)

