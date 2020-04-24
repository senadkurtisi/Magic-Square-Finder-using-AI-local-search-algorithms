import numpy as np

import random

class Magic:
	def __init__(self, n, magic = None):
		self.n = n
		self.target = int((n*(n**2 + 1))/2)

		if magic is None:
			self.magic = np.arange(start = 1, stop = n**2+1).astype(int)
			np.random.shuffle(self.magic)
			self.magic = (self.magic).reshape((n, n))
		else:
			self.magic = magic


	def __eq__(self, other):
		''' Two magic squares are treated as equal
			if their matrices are exactly the same.
		'''
		return np.array_equal(self.magic, other.magic)


	def set_magic(self, magic):
		''' Changes the magic array to the
			specified array.

		Arguments:
			magic (numpy.ndarray): the array to which
								   we wish to change
								   the current magic array
		'''
		self.magic = np.copy(magic)


	def number_of_violations(self):
		''' Calculates the number of violations, 
			or the number of rules of the 
			"magic square concept", by calculating
			the number of row, columns and diagonals
			which don't sum up to the target value
			defined with the class attribute: self.target

		Returns:
			violations (int): number of violations
		'''
		violations = 0

		# column sums & violations
		col_sums = np.sum(self.magic, axis = 0)
		violations += np.count_nonzero(col_sums != self.target)

		# row sums & violations
		row_sums = np.sum(self.magic, axis = 1)
		violations += np.count_nonzero(row_sums != self.target)

		# main diagonal
		main_diag_sum = np.trace(self.magic)
		violations += int(main_diag_sum != self.target)

		# secondary diagonal
		second_diag_sum = np.trace(np.fliplr(self.magic))
		violations += int(second_diag_sum != self.target)

		return violations


	def get_fitness(self):
		''' Returns fitness function needed for
			the Genetic Algorithm. Fitness fcn
			is number of satisfied constraints.
		'''
		return self.objective_fcn()/(1.5)


	def objective_fcn(self):
		''' Returns objective function we wish to
			maximize using simulated annealing approach.
			Obj. function is represented as number of
			satisfifed constraints.
		'''
		return ((self.n)*2 + 2 - self.number_of_violations())*1.5


	def swap_cells(self, first, second):
		''' Swaps elements of the cells at the
			specified positions.

		Arguments:
			first (tuple): 0-based indexes of the first
						   cell's position
			second (tuple): 0-based indexes of the second
						    cell's position
		'''
		first_row, first_col = first
		second_row, second_col = second

		self.magic[first_row, first_col], self.magic[second_row, second_col] = \
			self.magic[second_row, second_col], self.magic[first_row, first_col]


	def generate_descendants(self, max = 500):
		''' Generates the specified number of descendants
			by switching rows, columns, diagonals and cells
			with respect to maximum possible number of 
			descendats a magic square can have according to
			it's dimensions.

		Arguments:
			max (int): number of descendants to generate
		''' 
		descendants = []
		desc_swap = []

		start_arr = self.magic

		N = (self.n)**2
		real_max = ((N-1)*N)/2
		real_max += (self.n)*(self.n-1)
		real_max += 1

		# switching rows
		for row in range(self.n):
			for r in range(1, self.n):
				magic = np.copy(self.magic)
				magic[[row, r]] = magic[[r, row]]
				descendants.append(Magic(self.n, magic))

				if len(descendants)==max:
					return descendants

		# switching columns
		for col in range(self.n):
			for c in range(1, self.n):
				magic = np.copy(self.magic)
				magic[[col, c]] = magic[[c, col]]
				descendants.append(Magic(self.n, magic))

				if len(descendants)==max:
					return descendants

		# switching diagonals
		for ind in range(self.n):
			magic = np.copy(self.magic)
			magic[ind, ind], magic[ind, self.n-1-ind] = \
				magic[ind, self.n-1-ind], magic[ind, ind]
			descendants.append(Magic(self.n, magic))

			if len(descendants)==max:
				return descendants

		# swapping random cells
		for ind in range(max):
			found = False
			while not found:
				swap_x, swap_y = random.sample(range(N), 2)

				if not [swap_x, swap_y] in desc_swap:
					desc = (np.copy(start_arr)).flatten()
					desc_swap += [[swap_x, swap_y], [swap_y, swap_x]]
					desc[swap_x], desc[swap_y] = desc[swap_y], desc[swap_x]

					desc = desc.reshape((self.n, self.n))
					descendants.append(Magic(self.n, magic = desc))

					if len(descendants) == real_max:
						return descendants

					found = True


		return descendants
