import numpy as np
import matplotlib.pyplot as plt

from .magic import Magic

class HillClimbing:
	def __init__(self, magic, n = 5, iterations = 100, debug = False):
		self.iterations = iterations
		self.n = n
		self.magic = Magic(n)
		self.magic.set_magic(magic)

		self.path = []
		self.debug = debug


	def pick_next_move(self):
		''' Picks the next magic number solution
			using the hill climbing inference logic.
			The algorithm uses class attribute path
			which consists of all previous states
			in order to prevent infinite loops.

		Returns:
			desc (list): [0]: (Magic obj.) -- first POSSIBLE
												  magic square which
												  with the lowest
												  heuristics

						 [1]: (int) -- heuristics for 
						 			   the best possible
						 			   descendant

		'''
		target = 2*(self.n)+2
		descendants = self.magic.generate_descendants(1000)
		descendants.sort(key=lambda desc: (target-desc.number_of_violations()), reverse=True)


		for desc in descendants:
			desc_array = desc.magic

			is_valid = True
			for arr in self.path:
				if np.array_equal(desc_array, arr):
					is_valid = False

			if is_valid:
				return desc, desc.number_of_violations()

		return [None, -1]

	def find_solution(self):
		''' Finds the solution using the
			hill climbing algorithm.

		Returns:
			(list: [0] (Magic obj.) -- magic square solution
									   found after specified
									   number of iterations
									   or when target solution
									   was found.
				   [1] (int) -- number of violations for the
				   				best magic square found
				   [2] (int) -- the iteration at which the best
				   				magic square was found

		'''
		score = 0
		viol = []

		for i in range(self.iterations):
			self.magic, self.score = self.pick_next_move()
			viol.append(self.score)

			if self.magic is not None:
				# Check for target solution
				if self.score == 0:
					if self.debug:
						plt.figure()
						plt.plot(viol)
						title = "Number of violations for {}x{}".\
												format(self.n, self.n)
						plt.title(title, fontsize=9)
						plt.show()

					return [self.magic, 0, i+1]

				self.path.append(self.magic.magic)
			else:
				i -= 1

		if self.debug:
			plt.figure()
			plt.plot(viol)
			title = "Number of violations for {}x{}".format(self.n, self.n)
			plt.title(title, fontsize=9)
			plt.show()
		
		return [self.magic, (self.magic).number_of_violations(), self.iterations]
		