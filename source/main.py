import numpy as np

import matplotlib.pyplot as plt

from classes.magic import Magic
from classes.random_search import RandomSearch
from classes.hill_climbing import HillClimbing
from classes.simulated_annealing import SimulatedAnnealing
from classes.beam_search import BeamSearch
from classes.genetic_algorithm import GeneticAlgorithm


if __name__ == "__main__":
	plt.close('all')
	# Get desired dimension of the magic square
	n = int(input("Enter dimension of the magic square: "))

	# MONTE CARLO 
	# Random Search, Hill Climbing, Beam search and Simulated annealing
	results_rs = []
	iterations_rs = []
	results_hill = []
	iterations_hill = []
	results_bs = []
	iterations_bs = []
	results_sa = []
	iterations_sa = []

	for i in range(100):
		magic = Magic(n)

		# Random walk
		random_search = RandomSearch(n = n, iterations = 5000)
		best_magic_rs, best_viol_rs, iter_rs = random_search.generate_best()
		results_rs.append(best_viol_rs)
		iterations_rs.append(iter_rs)

		# Hill Climbing
		hill_climbing = HillClimbing(magic = magic.magic, n = n, iterations = 5000)
		best_magic_hc, best_score_hc, iter_hc = hill_climbing.find_solution()
		results_hill.append(best_score_hc)
		iterations_hill.append(iter_hc)

		# Beam search
		beam_search = BeamSearch(magic = magic.magic, n = n, iterations = 5000)
		best_magic_bs, best_score_bs, iter_bs = beam_search.generate_solution()
		results_bs.append(best_score_bs)
		iterations_bs.append(iter_bs)

		# Simulated annealing
		simul_annealing = SimulatedAnnealing(magic = magic.magic, n = n, iterations = 5000)
		best_magic_sa, best_score_sa, iter_sa = simul_annealing.get_solution()
		results_sa.append(best_score_sa)
		iterations_sa.append(iter_sa)

		print("End of iteration", i+1)


	print("Average number of violations using Random Search:", \
				sum(results_rs)/len(results_rs))
	print("Standard deviation of violations using Random Search:",\
				np.std(results_rs))
	print("Average number of iterations using Random Search:", \
				sum(iterations_rs)/len(iterations_rs))
	print("Standard deviation of iterations using Random Search:", \
				np.std(iterations_rs))


	print("Average number of violations using Hill Climbing:", \
				sum(results_hill)/len(results_hill))
	print("Standard deviation of violations using Hill Climbing:", \
				np.std(results_hill))
	print("Average number of iterations using Hill Climbing:", \
				sum(iterations_hill)/len(iterations_hill))
	print("Standard deviation of iterations using Hill Climbing:", \
				np.std(iterations_hill))


	print("\nAverage number of violations using Beam Search:", 
				sum(results_bs)/len(results_bs))
	print("Standard deviation of violations using Beam Search:", \
				np.std(results_bs))
	print("Average number of iterations using Beam Search:", \
				sum(iterations_bs)/len(iterations_bs))
	print("Standard deviation of iterations using Beam Search:", \
				np.std(iterations_bs))


	print("\nAverage number of violations using Simulated Annealing:", \
				sum(results_sa)/len(results_sa))
	print("Standard deviation of violations using Simulated Annealing:", \
				np.std(results_sa))
	print("Average number of iterations using Simulated Annealing:", \
				sum(iterations_sa)/len(iterations_sa))
	print("Standard deviation of iterations using Simulated Annealing:", \
				np.std(iterations_sa))


	# Genetic algorithm
	results_ga = []
	iterations_ga = []
	for i in range(100):
		gen_algorithm = GeneticAlgorithm(gen_size = 200, n = n, \
										iterations = 1000, debug = False)
		best_magic_ga, best_score_ga, iter_ga = gen_algorithm.get_best()
		print(best_score_ga)
		results_ga.append(best_score_ga)
		iterations_ga.append(iter_ga)

		print("End of iteration", i+1)

	print("\nAverage number of violations using Genetic Algorithm:", \
				sum(results_ga)/len(results_ga))
	print("Standard deviation of violations using Genetic Algorithm:", \
				np.std(results_ga))
	print("Average number of iterations using Genetic Algorithm:", \
				sum(iterations_ga)/len(iterations_ga))
	print("Standard deviation of iterations using Genetic Algorithm:", \
				np.std(iterations_ga))
	
	# Comparing genetic algorithm with beam search
	gen_algorithm = GeneticAlgorithm(gen_size = 200, n = 6, \
										iterations = 1000, debug = True)
	best_magic_ga, best_score_ga, iter_ga = gen_algorithm.get_best()

	beam_search = BeamSearch(magic = magic.magic, n = n, iterations = 5000, \
																 debug = True)
	best_magic_bs, best_score_bs, iter_bs = beam_search.generate_solution()

	plt.show()
