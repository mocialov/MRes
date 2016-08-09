import matplotlib.pyplot as plt
import numpy
import sys
from matplotlib import cm as CM

filename = 'best_solution_'+sys.argv[1]+'_generation_'+sys.argv[2]

with open(filename, 'r') as content_file:
    content = content_file.read()
alist=content.split(" || ")
fitness=alist[0]
fitness=fitness.split(": ")
fitness=fitness[1]

with open('fitness/'+filename, 'w') as file_:
    file_.write(fitness)
