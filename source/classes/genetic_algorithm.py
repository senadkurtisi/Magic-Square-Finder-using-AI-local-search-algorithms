import numpy as np
from decimal import *
import matplotlib.pyplot as plt

import random

import sys

from .magic import Magic

class GeneticAlgorithm:
	
	def __init__(self, n = 5, gen_size = 100, iterations = 1000, debug = False):
		self.n = n
		self.iterations = iterations

		self.gen_size = gen_size

		self.debug = debug


	def restore_child(self, child):
		''' Restores the child vector from given
			inversion sequence.

		Arguments:
			child (numpy.ndarray): inversion sequence of the child
								   created by crossing it's parents
		Returns:
			new_child (numpy.ndarray): New child formed by restoring
									   from it's inverse form
		'''
		N = child.size
		restored_child = (np.zeros(child.shape).astype(np.int16)).tolist()
		restored_child[-1] = child[N-1]

		for ind in range(N-2, -1, -1):
			restored_child[ind] = child[ind]
			bigger_equal = 0

			for k in range(ind+1, N):
				element = restored_child[k]
				if element >= restored_child[ind]:
					restored_child[k] += 1


		new_child = np.zeros(child.shape)
		for ind in range(N):
			bigger = restored_child[ind]
			new_child[bigger] = ind+1

		return new_child


	def get_inversion_sequence(self, parent):
		''' Creates the inversion sequence of the given
			parent. The inv. seq. is created in such way
			that from numbers i: 1-n^2 we count the number
			of numbers greater than i in the parent vector.
			Then we store the count in the inverse vector
			at the position i-1.

		Arguments:
			parent (numpy.ndarray): parent vector - original
		Returns:
			inversion (numpy.ndarray): inverse sequence of the
									   of the given parent vector
		'''
		inversion = np.zeros(parent.shape).astype(np.int16)
		parent = parent.tolist()

		for i in range(inversion.size):
			larger = 0
			for j in parent:
				if j==(i+1):
					break
				else:
					if j>(i+1):
						larger += 1
			inversion[i] = larger

		return inversion


	def crossover(self, par_x, par_y):
		''' Performs crossover operator on the given
			pair of parents. The parents are reshaped
			into a vector stacked row next to row.
			Crossover point is generated using single
			point crossover and after that we cross
			the inversion sequences of the parents
			to form two inversion sequences of their
			children which we restore right after.

		Arguments:
			par_x (numpy.ndarray): first parent
			par_y (numpy.ndarray): second parent
		Returns:
			child_A (Magic obj.): Magic square corresponding
								  to the first child
			child_B (Magic obj.): Magic square corresponding
								  to the second child
		'''
		N = np.size(par_x.magic)
		parent_x = (np.copy(par_x.magic)).flatten()
		parent_y = (np.copy(par_y.magic)).flatten()

		pos = pos = random.randint(0, N-1)
		inv_x = self.get_inversion_sequence(parent_x)	
		inv_y = self.get_inversion_sequence(parent_y)

		child_A = np.concatenate((inv_x[:pos], inv_y[pos:]))
		child_B = np.concatenate((inv_y[:pos], inv_x[pos:]))		

		child_A = (self.restore_child(child_A)).reshape((self.n, self.n))
		child_B = (self.restore_child(child_B)).reshape((self.n, self.n))

		return [Magic(self.n, child_A), Magic(self.n, child_B)]


	def get_best(self):
		''' Returns the best Magic square found using
			Genetic Algorithm. GA uses elitism that
			consists of the best 10% of the population.

		Returns:
			(list): [0] -- best magic square found
					[1] -- number of violations for that
						   magic square
					[2] -- iteration where the best magic
						   square was found
		'''
		MATING_POOL_SIZE = round(self.gen_size/2)
		ELITE_SIZE = round(0.1*self.gen_size)
		MUTATION_P = 0.1

		generation = []
		average_viol = []
		minimum_viol = []
		viol = []

		# creating the initial population
		for i in range(self.gen_size):
			found = False
			while not found:
				found = True
				new_member = Magic(self.n)
				if generation:
					if new_member in generation:
						found = False

			generation.append(new_member)


		for it in range(self.iterations):
			viol = [member.number_of_violations() for member in generation]
			average_viol.append(sum(viol)/len(viol))

			# Checking for an ideal solution
			generation.sort(key = lambda x: x.number_of_violations(), reverse = False)
			minimum_viol.append(generation[0].number_of_violations())
			if generation[0].number_of_violations() == 0:
				if self.debug:
					print("SOLUTION FOUND ::: ITERATION", it)

				return [generation[0], 0, it]


			new_generation = []
			parents = []

			random.shuffle(generation)

			# Calculating the fitness function for the entire generation
			fitness = [x.get_fitness() for x in generation]
			# Calculating the probability for picking each member of the gen.
			prob = [fit/sum(fitness) for fit in fitness]

			for i in range(MATING_POOL_SIZE):
				while True:
					# Roulette wheel selection of the parents
					parent_x, parent_y = np.random.choice(generation, 2, p = prob)

					# testing the edge cases
					if ((parent_x, parent_y) in parents) or ((parent_y, parent_x) in parents):
						continue
					if parent_x == parent_y:
						continue

					parents.append((parent_x, parent_y))
					break

			for i in range(MATING_POOL_SIZE):
				# Forming children with crossover
				children = self.crossover(*(parents[i]))

				# Mutation
				for child in children:
					mutate = np.random.choice([True, False], p = [MUTATION_P, 1-MUTATION_P])

					# Mutation randomly swaps two fields
					if mutate:
						# Randomly pick fields to mutate
						first, second = random.sample(range((self.n)**2), 2)

						current_magic = (np.copy(child.magic)).flatten()
						current_magic[first], current_magic[second] = \
							current_magic[second], current_magic[first]
						current_magic = current_magic.reshape((self.n, self.n))

						child.set_magic(current_magic)

				new_generation += children
	

			# Transfering the elite to the next generation
			generation.sort(key = lambda x: x.get_fitness(), reverse = True)
			generation = generation[:ELITE_SIZE]

			# Completing the next generation
			generation += new_generation
			generation.sort(key = lambda x: x.get_fitness(), reverse = True)
			generation = generation[:self.gen_size]


			if (it+1)%50==0:
				if self.debug:
					print("End of iteration", it+1)
					print([elem.number_of_violations() for elem in generation[:5]])


		if self.debug:
			fig, ax = plt.subplots(nrows = 2, ncols = 1)

			ax[0].plot(average_viol)
			ax[0].set_title("Average number of violations")

			ax[1].plot(minimum_viol)
			ax[1].set_title("Minimum number of violations")

			plt.show()

		generation.sort(key = lambda x: x.number_of_violations(), reverse = False)
		return [generation[0], (generation[0]).number_of_violations(), self.iterations]





