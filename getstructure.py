import pickle
import MultiNEAT as NEAT
import visualize_plot as viz
import cv2
import matplotlib.pyplot as plt
import numpy


best_genome_from_all = None
filename = 'best_from_gen_1960_fit_0.957201086957'

best_genome_from_all = NEAT.Genome(filename)
# net = NEAT.NeuralNetwork()
# best_genome_from_all = NEAT.BuildPhenotype(net)
# best_genome_from_all = NEAT.Load(filename)
# best_genome_from_all = pickle.load(open( 'winner-feedforward (7).pickle', "rb" ))

# with open('bestie', "r") as f:
#      best_genome_from_all = pickle.load(f)

viz.Draw(best_genome_from_all)