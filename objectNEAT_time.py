from __future__ import print_function
import os
import sys
import csv
import time
import MultiNEAT as NEAT

import random
import numpy as np
import pickle as pickle
import load_images_2classRGB

import cv2

# import MultiNEAT.viz as viz
import visualize_plot as viz
import matplotlib.pyplot as plt
def get_label(onehot):
	for c in range(len(onehot)):
		if onehot[c]==1:
			return c

def get_max(softmax):
	max_soft = np.max(softmax)
	place = 0
	i = -1
	for c in softmax:
		i += 1
		if c == max_soft:
			return c , i
def filename_fix_existing(filename):
	"""Expands name portion of filename with numeric ' (x)' suffix to
	return filename that doesn't exist already.
	"""
	dirname = '.' 
	name, ext = filename.rsplit('.', 1)
	names = [x for x in os.listdir(dirname) if x.startswith(name)]
	names = [x.rsplit('.', 1)[0] for x in names]
	suffixes = [x.replace(name, '') for x in names]
	# filter suffixes that match ' (x)' pattern
	suffixes = [x[2:-1] for x in suffixes
				   if x.startswith(' (') and x.endswith(')')]
	indexes  = [int(x) for x in suffixes
				   if set(x) <= set('0123456789')]
	idx = 1
	if indexes:
		idx += sorted(indexes)[-1]
	return '%s (%d).%s' % (name, idx, ext)

def run_objectrecognition(number_of_generations, save_path, test_size):

	data_filename_all = 'genration_data_all.csv'
	data_filename_all = os.path.join(save_path, data_filename_all)

	data_filename_stats = 'generation_data_stats.csv'
	data_filename_stats = os.path.join(save_path, data_filename_stats)

	data_filename_test = 'data_filename_test.csv'
	data_filename_test = os.path.join(save_path, data_filename_test)

	dual_fit = 'duals_fitnesses.csv'
	dual_fit = os.path.join(save_path, dual_fit)

	average_imagename = 'average_std.png'
	average_imagename = os.path.join(save_path, average_imagename)

	individuals_imagename = 'individuals.png'
	individuals_imagename = os.path.join(save_path, individuals_imagename)

	filename_winner = 'winner-feedforward.pickle'
	filename_winner = os.path.join(save_path, filename_winner)

	network_structure_filename = 'best_network_gen_'
	network_structure_filename = os.path.join(save_path, network_structure_filename)

	test_inputs_filename = 'test_inputs'
	test_inputs_filename = os.path.join(save_path, test_inputs_filename)

	test_outputs_filename = 'test_outputs'
	test_outputs_filename = os.path.join(save_path, test_outputs_filename)

	best_network_filename = 'winner_network_with_save.txt'
	best_network_filename = os.path.join(save_path, best_network_filename)

	specie_filename = 'speciation.csv'
	specie_filename = os.path.join(save_path, specie_filename)

	parameters = 'parameters.txt'
	parameters = os.path.join(save_path, parameters)

	plot_filename = 'best_network_gen_'
	plot_filename = os.path.join(save_path, plot_filename)

	start_time_total = time.time()
	train_inputs, test_inputs , train_outputs, test_outputs = load_images_2classRGB.load_data(test_size)

	with open(test_inputs_filename, 'wb') as f:
		pickle.dump(test_inputs, f)

	with open(test_outputs_filename, 'wb') as f:
		pickle.dump(test_outputs, f)


	params = NEAT.Parameters() 
	params.PopulationSize = 100
	params.MinSpecies = 5
	params.MaxSpecies = 15
	params.CompatTreshold = 0.3

	params.OverallMutationRate = 0.75

	params.MutateWeightsProb = 0.7
	params.WeightMutationRate = 0.4
	params.WeightMutationMaxPower = 0.5
	params.MutateWeightsSevereProb = 0.05

	params.WeightReplacementMaxPower = 4.0
	params.WeightReplacementRate = 0.2
	

	params.MutateAddNeuronProb = 0.1
	params.MutateAddLinkProb = 0.1
	params.MutateRemLinkProb = 0.01

	

	num_inputs = 1110 + 1 # number of inputs. Note: always add one extra input, for bias
	num_hidden_nodes = 0
	num_outputs = 2
	output_activation_function = NEAT.ActivationFunction.UNSIGNED_SIGMOID
	hidden_activation_function = NEAT.ActivationFunction.UNSIGNED_SIGMOID

	params.Save(parameters)

	genome = NEAT.Genome(
			0,
			num_inputs,
			num_hidden_nodes,
			num_outputs,
			False,  # FS_NEAT; auto-determine an appropriate set of inputs for the evolved networks
			output_activation_function,
			hidden_activation_function,
			0,
			params,
			0
	)


	pop = NEAT.Population(
		genome,
		params,
		True,  # whether the population should be randomized
		3.0,  # how much the population should be randomized,
		50 # 20 papaer(2)# 10 (paper 1) # the 42 is the RNG seed #41,42,
	)
	def softmax2(array):
		return np.exp(array) / np.sum(np.exp(array), axis=0)

	def evaluate(genome, trainx , trainy):
		# this creates a neural network (phenotype) from the genome
		net = NEAT.NeuralNetwork()
		genome.BuildPhenotype(net)
		empty = [0,0]
		genome_fitness = 0
		genome_time = time.time()
		cnt = 0
		for _im, _label in zip(trainx, trainy):
			net.Input( _im )
			net.Activate()
			output = net.Output()
			output = softmax2(output)
			dotproduct = np.dot(np.array(output),np.array(_label))
			if dotproduct > 0.5:
				bonus = abs(np.array(output)[0]-np.array(output)[1])
				genome_fitness += dotproduct + bonus
				cnt += 1
			else:
				bonus = abs(np.array(output)[0]-np.array(output)[1])
				genome_fitness -= ((1-dotproduct) + bonus)

		genome_time_end = float((time.time() - genome_time))

		# print('mean genome', np.mean(empty))
		# print('accur genome', np.sum(empty)/float(len(trainy)))
		# print('std genome', np.std(empty))
		# print('1/1+std genome', 1.0/(1+np.std(empty)))

		fitness_images = genome_fitness / (2.0*float(len(trainy)))
		# print(evaluate_time_end)
		
		# try 0.01, 0.1, 0.05
		# print(fitness_images)
		fitness = fitness_images
		# the output can be used as any other Python iterable. For the purposes of the tutorial, 
		# we will consider the fitness of the individual to be the neural network that outputs constantly 
		# 0.0 from the first output (the second output is ignored)
		return fitness, genome_time_end, cnt

	all_fitness_per_generation = []
	std_fitness_per_generation = []
	avg_fitness_per_generation = []
	best_fitness_per_generation = []
	intermediate_test_fitness_list = []
	specie_list = []
	gens = []
	test_gens = []
	best_genome_ever = None
	best_genome_ever_fitness = 0
	all_dual_fitnesses = []
	for generation in range(number_of_generations):
		
		# if generation%50 == 0:
		# 	m += 100
		# 	print(('----Number of samples: {!r}'.format(m)))
		# 	if m < len(train_outputs):
		x_sub, y_sub = train_inputs, train_outputs

			# else:
			# 	x_sub, y_sub = train_inputs, train_outputs

		# if generation == 1500:
		# 	params.MutateWeightsProb = 0.8
		# 	params.WeightMutationMaxPower = 2.0
		# 	params.MutateAddNeuronProb = 0.1
		# 	params.MutateAddLinkProb = 0.1

		# if generation == 2500:
		# 	params.WeightMutationMaxPower = 0.4
		# 	params.MutateWeightsProb = 0.9

		start_time = time.time() # run for 100 generations
		gens.append(generation)
		print('---------------------------------------------------------------------')
		print('---- Generation: {!r}'.format(generation))
		print('---------------------------------------------------------------------')


		# retrieve a list of all genomes in the population
		genome_list = NEAT.GetGenomeList(pop)
		print('individuals:')
		print(len(genome_list))
		fitnesslist = []
		time_fitness_list = []
		count_list_good = []
		count_list_final = []
		total_fitness = 0
		best = 0
		best_genome = None
		
		# apply the evaluation function to all genomes
		for genome in genome_list:
			fitness, time_fitness,cnt_ = evaluate(genome, x_sub, y_sub)
			fitnesslist.append(fitness)
			time_fitness_list.append(time_fitness/float(len(y_sub)))
			count_list_good.append(cnt_)
			total_fitness += fitness

			if generation % 100 == 0:
				x_sub_test, y_sub_test = test_inputs, test_outputs
				fitness, time_fitness,cnt_fin = evaluate(genome, x_sub_test, y_sub_test)
				count_list_final.append(cnt_fin)

		max_time_fit = np.max(time_fitness_list)
		min_time_fit = np.min(time_fitness_list)

		print(time_fitness_list)
		print(min_time_fit, max_time_fit)
		# normalize_list = (time_fitness_list - min_time_fit)/(max_time_fit - min_time_fit)
		normalize_list = (time_fitness_list)/np.sum(time_fitness_list)
		# normalize_list = list(normalize_list)
		all_dual_fitnesses.append([fitnesslist, list(0.001/normalize_list), count_list_good,count_list_final]) # previous: 0.001/timefit
		print('normalized list: ', normalize_list)
		for genome, fitn, timefit in zip(genome_list, fitnesslist, normalize_list):
			end_fitness = fitn + 0.0001/timefit
			print('end fitness: ')
			print(fitn,'+', 0.0001/timefit, '=', end_fitness)
			
			genome.SetFitness(end_fitness)
			if end_fitness > best:
				best_genome = genome
				if end_fitness > best_genome_ever_fitness:
					best_genome_ever = genome
			else:
				best_genome = best_genome
		if generation % 10 == 0:
			best_genome_from_gen = pop.GetBestGenome()
			net = NEAT.NeuralNetwork()
			best_genome_from_gen.BuildPhenotype(net)
			best_genome_from_gen.Save(best_network_filename+'_intermediate_'+str(generation))
			if generation % 400 == 0:
				viz.Draw(best_genome_from_gen)
				fig = plt.gcf()
				plt.axis([-100, 4100, -100, 4100])
				plt.axis('off')
				fig.savefig(plot_filename+'_gen_'+str(generation)+'.png', dpi = 150, transparent = True)


			intermediate_test_fitness = 0
			pos_test_set_good = 0
			neg_test_set_good = 0
			test_gens.append(generation)
			for _im, _label in zip(test_inputs, test_outputs):
				net.Input( _im )
				net.Activate()
				output = net.Output()

				output_list = [output[0]] + [output[1]]
				output = softmax2(output)
				dotproduct = np.dot(np.array(output),np.array(_label))

				if dotproduct > 0.5:
					intermediate_test_fitness += 1
					if _label[0] == 1:
						pos_test_set_good += 1
					elif _label[1] == 1:
						neg_test_set_good += 1
				else:
					intermediate_test_fitness += 0

			intermediate_test_fitness = intermediate_test_fitness / float(len(test_outputs))
			intermediate_test_fitness_list.append(intermediate_test_fitness)
			print('---------------------------------------------------------------------')
			print('Number of positive samples correctly classified: ', pos_test_set_good)
			print('Number of negative samples correctly classified: ', neg_test_set_good)
			print('---------------------------------------------------------------------')
			print('--- Intermediate test fitness for generation {!r}:  {!r}--------------'.format(generation, intermediate_test_fitness))
			print('---------------------------------------------------------------------')


		print("--- Evolving time:             {!r} seconds ---".format((time.time() - start_time)))

		count_list = []
		for s in pop.Species:
			# print('specie:')
			# print(s)
			count = 0
			for i in s.Individuals:
				
				count+= 1
			# print('individuals:')
			# print(count)	
			row =[generation, s.ID(), s.GetLeader().GetFitness(), count]
			print(generation, s.ID(), s.GetLeader().GetFitness(), count)
			specie_list.append(row)


		all_fitness_per_generation.append(fitnesslist)
		avg_fitness = total_fitness / float(len(genome_list))
		print('--- Number of species:           {!r}--- '.format(len(pop.Species)))
		avg_fitness_per_generation.append(avg_fitness)
		print('--- Average fitness:             {!r}--- '.format(avg_fitness))

		std_fitness = np.std(fitnesslist)
		std_fitness_per_generation.append(std_fitness)
		print('--- Standard deviation fitness:  {!r}--- '.format(std_fitness))

		best_fitness = max(fitnesslist)
		best_fitness_per_generation.append(best_fitness)
		print('--- Best fitness:                {!r}--- '.format(best_fitness))

		# at this point we may output some information regarding the progress of evolution, best fitness, etc.
		# it's also the place to put any code that tracks the progress and saves the best genome or the entire
		# population. We skip all of this in the tutorial. 


		# advance to the next generation
		print("--- Generation time:             {!r} seconds ---".format((time.time() - start_time)))
		pop.Epoch()

	best_genome_from_all = pop.GetBestGenome()
	print(best_genome_from_all)
	print(best_genome_ever)
	net = NEAT.NeuralNetwork()
	best_genome_from_all.BuildPhenotype(net)
	test_fitness = 0

	viz.Draw(best_genome_from_all)
	fig = plt.gcf()
	plt.axis([-100, 4100, -100, 4100])
	plt.axis('off')
	fig.savefig(plot_filename+str(best_genome_from_all)+'.png', dpi = 150, transparent = True)

	best_genome_from_all.Save(best_network_filename)

	with open(filename_winner, 'wb') as f:
			pickle.dump(best_genome_from_all, f)

	pos_test_set_good = 0
	neg_test_set_good = 0
	empty_final = [0,0,0,0]
	for _im, _label in zip(test_inputs, test_outputs):
		net.Input( _im )
		net.Activate()
		output = net.Output()
		output_list = [output[0]] + [output[1]]
		output = softmax2(output)
		dotproduct = np.dot(np.array(output),np.array(_label))
		if dotproduct > 0.5:
			test_fitness += 1
			if _label[0] == 1:
				pos_test_set_good += 1
			elif _label[1] == 1:
				neg_test_set_good += 1
		# else:
		# 	_im = np.reshape(_im[:(len(_im)-1)], (8, 37))
		# 	plt.imshow(_im)
		# 	plt.title(str(_label))
		# 	plt.show()
		# 	test_fitness += 0


	test_fitness_total = test_fitness / (float(len(test_outputs)))
	print('test fitness:{!s}'.format(test_fitness_total))
	print('pos_tes_sample_correct:{!s}'.format(pos_test_set_good))
	print('neg_tes_sample_correct:{!s}'.format(neg_test_set_good))
	print('total test smaples:{!s}'.format(len(test_outputs)))
	


	# add: size of network, save network,, species


	with open(data_filename_all, 'w') as f:
		w = csv.writer(f, delimiter=',')
		for s in all_fitness_per_generation:
					w.writerow(s)

	with open(specie_filename, 'w') as f:
		w = csv.writer(f, delimiter=',')
		for s in specie_list:
					w.writerow(s)

	with open(data_filename_test, 'w') as f:
		w = csv.writer(f, delimiter=',')
		for s in intermediate_test_fitness_list:
					w.writerow([s])

	with open(data_filename_stats, 'w') as f:
		w = csv.writer(f, delimiter=',')
		for avg, std, best in zip(avg_fitness_per_generation, std_fitness_per_generation, best_fitness_per_generation):
					w.writerow([avg, std, best])

	with open(dual_fit, 'w') as f:
			w = csv.writer(f, delimiter=',')
			for s in all_dual_fitnesses:
					w.writerow(s)
	
	print('_____________________________________________________________________')
	print('---------------------------------------------------------------------')
	print("--- Total time for {!r} generations: {!r} minutes ---".format(len(gens),(time.time() - start_time_total)/60.0))


	plt.figure(figsize=( 7.195,3.841), dpi=150)
	plt.scatter(gens, avg_fitness_per_generation, s=0.1 , c='green')
	plt.scatter(gens, np.array(avg_fitness_per_generation) + np.array(std_fitness_per_generation), s=0.1 , c='red')
	plt.scatter(gens, np.array(avg_fitness_per_generation) - np.array(std_fitness_per_generation), s=0.1 , c='red')
	plt.scatter(gens, best_fitness_per_generation, s=0.1 , c='blue')
	plt.plot(test_gens, intermediate_test_fitness_list,linewidth=1.0, c='blue')
	plt.ylabel("Average fitness")
	plt.xlabel("Generations")
	plt.savefig(average_imagename)
	# plt.show()

	plt.figure(figsize=( 7.195,3.841), dpi=150)
	for xe, ye in zip(gens, all_fitness_per_generation):
			plt.scatter([xe] * len(ye), ye, s=0.01 , c='green')

	plt.ylabel("Fitness per individual")
	plt.xlabel("Generations")
		# plt.axes().set_xticklabels(generation)
		
	plt.savefig(individuals_imagename)

	# plt.show()