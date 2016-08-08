#!/usr/bin/python

# Python experiment setup for peas HyperNEAT implementation
# This setup is used by Webots to perform evolution of a controller for e-puck robot that tries to follow a line created as a 'world' in Webots environment

# Program name - line_following_webots.py
# Written by - Boris Mocialov (bm4@hw.ac.uk)
# Date and version No:  20.05.2015

# Webots supervisor class spawns this script, which is used to evolve a controller that is evaluated in Webots environment.
# The interaction between Webots environment and peas implementation is done via '.dat' files.
# peas implementation is only provided with the fitness of each generated controller - peas does no evaluation itself

# Fitness function is defined in Webots environment

### IMPORTS ###
import sys, os
from functools import partial
from itertools import product

# Libs
import numpy as np
np.seterr(invalid='raise')

# Local
sys.path.append(os.path.join(os.path.split(__file__)[0],'..','..')) 
from peas.methods.neat import NEATPopulation, NEATGenotype
from peas.methods.evolution import SimplePopulation
from peas.methods.wavelets import WaveletGenotype, WaveletDeveloper
from peas.methods.hyperneat import HyperNEATDeveloper, Substrate
from peas.tasks.linefollowing import LineFollowingTask
from peas.tasks.targetweights import TargetWeightsTask

import time

import signal

#import admin

developer = None

class nonlocal:
    counter = 1

class Communication:
    PROVIDE_ANN = 1
    EVALUATION_RESULTS = 2

stats = {'fitness':0.0, 'dist':0, 'speed':0, 'nodes':0};

comm_direction = Communication.PROVIDE_ANN;

def pass_ann(node_types, ann, genes_file_number):
    #dir = os.path.dirname('C:/Users/ifxboris/Desktop/hwu/Webots/controllers/advanced_genetic_algorithm_supervisor/fifofile')
    #if not os.path.exists(dir):
    #    os.makedirs(dir)

    myfile = os.path.join(os.path.dirname(__file__), '../../genes_'+str(genes_file_number)+'gen_'+str(nonlocal.counter))
    while True:
        #print "pass ann"
        try:
            #print "trying"
            with open(myfile, 'w') as file: # automatically closes...
                file.write(' '.join(map(str, node_types)) + ' '+' '.join(map(str, ann))+'\nasd')
        except IOError, e:
            #print "exception"
            time.sleep(1)
            continue
        break # the while loop

    #fifo = open(os.path.join(os.path.dirname(__file__), '../../genes_bak_5'), 'wb')
    #fifo.write(' '.join(map(str, node_types)) + '\n'+' '.join(map(str, ann)))  #str(len(ann)) + ' ' + ' '.join(map(str, ann))
    #fifo.close()

    #print 'connectivity matrix length: '
    #print len(ann)

    comm_direction = Communication.EVALUATION_RESULTS
	
def get_stats(file_name):
    while not os.path.exists(os.path.join(os.path.dirname(__file__), '../../genes_fitness_'+str(file_name)+'gen_'+str(nonlocal.counter))):
            #print "file: ", (os.path.join(os.path.dirname(__file__), '../../genes_fitness_'+str(file_name)+'gen_'+str(nonlocal.counter)))
            #print "get_stats wait"
	    time.sleep(1)
	    continue

    #fileslist = os.listdir(".")
    #print fileslist
    #while (str(fileslist).count('gen_'+str(nonlocal.counter)) < 50):
    #        print "get_stats this thing"
    #	    time.sleep(1)
    #        fileslist = os.listdir(".")
    #        #print str(fileslist).count('gen_'+str(nonlocal.counter)), fileslist
    #	    continue

    #print "continuing"
    fitness = open(os.path.join(os.path.dirname(__file__), '../../genes_fitness_'+str(file_name)+'gen_'+str(nonlocal.counter)))
    comm_direction = Communication.PROVIDE_ANN
    fitness_data = fitness.read()
    fitness.close()
    time.sleep(5)
    os.remove(os.path.join(os.path.dirname(__file__), '../../genes_fitness_'+str(file_name)+'gen_'+str(nonlocal.counter)))
    nonlocal.counter += 1
    return fitness_data

def evaluate(individual, task, developer, file_name):
    #many evaluations, single solve
    #print 'evaluate'
    #print individual.get_weights() # <-- every individual weight matrix
    
    phenotype = developer.convert(individual)
    #print individual.get_network_data()
    #print '\n'
    #print phenotype.get_full_connectivity_matrix()
    #print 'result: '

    #print phenotype.get_node_types()

    nodes_types_indexes = list()
    #j=0
    #print 'length: '+str(len(phenotype.get_full_connectivity_matrix()))

    #print 'check this'
    #print len(phenotype.get_full_connectivity_matrix())
    #print 'and this'
    #nonlocal.counter += 1
    #phenotype.visualize('temp'+str(nonlocal.counter)+'.png', inputs=10, outputs=2)
    #print individual.get_network_data()[1]
    #print individual.get_network_data()[0]

    #for i in range(len(phenotype.get_full_connectivity_matrix())):
    #    if(np.count_nonzero(phenotype.get_full_connectivity_matrix()[:,i]) > 0): #if at least one connection from node (i)
    #    	print 'inferring node '+str(i)+ ' is connected'
        	#nodes_types_indexes.append(float(['sin', 'bound', 'linear', 'gauss', 'sigmoid', 'abs'].index((individual.get_network_data()[1])[j])))
        	#j += 1
        #else:
        	#nodes_types_indexes.append(float(['sin', 'bound', 'linear', 'gauss', 'sigmoid', 'abs'].index('sigmoid')))

    #print nodes_types_indexes

    rest = 101 - len((individual.get_network_data()[1])[1:])
    rest_array = np.linspace(4., 4., rest)

    #print rest_array

    for idx, node_type in enumerate((individual.get_network_data()[1])[1:]):
        if (idx == len((individual.get_network_data()[1])[1:]) - 2):
        	nodes_types_indexes.extend(rest_array)
        nodes_types_indexes.append(float(['sigmoid', 'sin', 'bound', 'linear', 'gauss', 'tanh', 'abs'].index(node_type)))

    pass_ann(nodes_types_indexes, phenotype.get_connectivity_matrix(), file_name) #phenotype.get_connectivity_matrix()
    fitness = get_stats(file_name)

    #print fitness

    #sys.exit()

    fitness = float('0'+fitness)
	
    #stats = task.evaluate(phenotype)
    print "fitness: ", fitness
    
    try:
        stats = {'fitness':float(fitness), 'dist':0, 'speed':0}  #what to do with dist and speed?
    except ValueError,e:
        stats = {'fitness':0.0001, 'dist':0, 'speed':0}  #what to do with dist and speed?

    
    if isinstance(individual, NEATGenotype):
        stats['nodes'] = len(individual.node_genes)
    elif isinstance(individual, WaveletGenotype):
        stats['nodes'] = sum(len(w) for w in individual.wavelets)
    #print '~',
    sys.stdout.flush()

    #print 'detector '+str(file_name)

    return stats
    
def solve(individual, task, developer):
    #phenotype = developer.convert(individual)
    #many evaluations, single solve
    #print 'solve'
    #phenotype = developer.convert(individual)
    #return int(self.evaluate(network) > 0.7) #task.solve(phenotype)
    #return task.solve(phenotype)
    #print 'solve'
    #print stats['fitness']
    #return stats['fitness'] > 0.7
    #print 'fitness solve:'
    #print individual.stats['fitness']
    return individual.stats['fitness'] > 0.05

def a_callback(self):
	print 'callback'


### SETUPS ###    
def run(method, setup, genes_file_number, generations=1000, popsize=50):

    shape = (100,1)
    task = TargetWeightsTask(substrate_shape=shape, fitnessmeasure='asdsadsadsad')
    
    #substrate = Substrate()
    #substrate.add_nodes(shape, 'l')
    #substrate.add_connections('l', 'l')

    substrate = Substrate()
    substrate.add_nodes([(r, theta) for r in np.linspace(-10,10,10) for theta in np.linspace(-10, 10, 10)], 'input', is_input=True)
    substrate.add_nodes([(r, theta) for r in np.linspace(0,0,1) for theta in np.linspace(0, 0, 1)], 'output')
    substrate.add_connections('input', 'output', -1)

    geno = lambda: NEATGenotype(feedforward=True, inputs=100, outputs=1, max_nodes=101, weight_range=(-1.0, 1.0), 
                                   prob_add_conn=0.8, prob_disable_conn=0.02, prob_add_node=0.25, prob_mutate_weight=0.1, stdev_mutate_weight=0.5,
                                   types=['tanh'])
    
    pop = NEATPopulation(geno, popsize=popsize, target_species=10)
    developer = HyperNEATDeveloper(substrate=substrate, add_deltas=True, sandwich=False)


    results = pop.epoch(developer, genes_file_number, generations=generations,
                        evaluator=partial(evaluate, task=task, developer=developer, file_name=genes_file_number),
                        solution=partial(solve, task=task, developer=developer), 
                        )


    fitnesses = list()

    fifo = open(os.path.join(os.path.dirname(__file__), '../../very_best_solution_'+genes_file_number), 'a+')
    for champion in results['champions']:
        fitnesses.append(champion.stats['fitness'])
        phenotype = developer.convert(champion)
        fifo.write('fitness: '+str(champion.stats['fitness'])+' || ' +' '.join(map(str, phenotype.get_connectivity_matrix()))+'\n')
    fifo.close()


    return results

if __name__ == '__main__':
    print 'running peas line following + webots'
    print sys.argv
    parent = sys.argv[1]
    resnhn = run('nhn', 'hard', sys.argv[2])
    print 'Done'
    #os.kill(parent, signal.SIGKILL)
