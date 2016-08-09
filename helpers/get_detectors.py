import matplotlib.pyplot as plt
import numpy
import sys
from matplotlib import cm as CM

filename = 'best_solution_'+sys.argv[1]+'_generation_'+sys.argv[2]

x = [1, 2, 3, 4, 5, 6, 7, 8]
y = [1, 2, 3, 4, 5, 6, 7, 8]

with open(filename, 'r') as content_file:
    content = content_file.read()
alist=content.split(" || ")
detector=alist[1].split()

for _ in range(101):
    detector.insert(0,4.0)

with open('detectors/detector_'+str(sys.argv[1]), 'w') as file_:
    for item in detector:
        file_.write(str(item)+" ")
