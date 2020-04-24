import numpy as np

from .magic import Magic

class RandomSearch:
	def __init__(self, n = 5, iterations = 100):
		self.iterations = iterations
		self.n = n

	def generate_best(self):
		''' Generates best solution using
			random walk. 

		Returns:
			(list): [0]: (Magic obj.) --  instance of Magic class
									 	  with the smallest amount
									 	  of violations.
					[1]: (int) -- number of violations for best_magic
					[2]: (int) -- iteration at which the best solution
								  was found
		'''
		best_magic = None
		best_score = np.inf

		for it in range(self.iterations):
			curr_magic = Magic(self.n)
			curr_score = curr_magic.number_of_violations()

			# Check for an ideal solution
			if curr_score == 0:
				return [curr_magic, 0, it+1]

			# Check for a better solution
			if curr_score<best_score:
				best_magic = curr_magic
				best_score = curr_score

		return [best_magic, best_score, self.iterations]

