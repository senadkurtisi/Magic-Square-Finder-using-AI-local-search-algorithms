import numpy as np
import matplotlib.pyplot as plt

from .magic import Magic


class BeamSearch:
	def __init__(self, magic, n = 5, iterations = 100, debug = False):
		self.n = n
		self.iterations = iterations

		self.magic = Magic(n)
		self.magic.set_magic(magic)

		self.path = []

		self.debug = debug


	def take_beam(self, current_beam, beam_size = 3):
		''' Takes the beam of the best k descendants
			of the given previouse beam. It works
			it's way around selecting the states that
			have already been selected.

		Arguments:
			current_beam (list): consists of the k best 
								 magic squares
			beam_size (int): represents the size of the beam
		Returns:
			beam (list): [0]-[k-1] -- k best magic squares out
									  of all the descendants of
									  the previous beam.
		'''
		descendants = []
		for node in current_beam:
			node_desc = (node).generate_descendants(500)

			descendants += node_desc

		beam = []
		descendants.sort(key=lambda desc: desc.number_of_violations(), reverse=False)

		for desc in descendants:
			is_valid = True
			for history in self.path:
				if np.array_equal(desc.magic, history.magic):
					is_valid = False

			if is_valid:
				beam.append(desc)
				if len(beam) == beam_size:
					return beam

		return None


	def generate_solution(self):
		''' Returns the best solution found using
			beam search. If the ideal solution is
			found earlier, it is immediately returned.

		Returns:
			(list): [0] -- best magic square found
					[1] -- number of violations for
						   the best magic square
					[2] -- iteration at which the best
						   magic square was found
		'''
		BEAM_SIZE = 3
		beam = [self.magic]

		average_viol = []
		minimum_viol = []
		viol = []

		for it in range(self.iterations):
			if it>0:
				viol = [node.number_of_violations() for node in beam]
				average_viol.append(sum(viol)/len(viol))
				viol.sort()
				minimum_viol.append(viol[0])

				self.path += beam

			beam = self.take_beam(beam, BEAM_SIZE)
			if beam is not None:
				beam.sort(key=lambda desc: desc.number_of_violations(), reverse=False)

				if (beam[0]).number_of_violations() == 0:
					return [beam[0], 0, it+1]
			else:
				it -= 1


		if self.debug:
			fig, ax = plt.subplots(nrows = 2, ncols = 1)

			ax[0].plot(average_viol)
			ax[0].set_title("Average number of violations")

			ax[1].plot(minimum_viol)
			ax[1].set_title("Minimum number of violations")

			plt.show()

		return [beam[0], beam[0].number_of_violations(), self.iterations]
