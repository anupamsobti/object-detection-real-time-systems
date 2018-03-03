#!/usr/bin/python3
import numpy as np
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser(description = "This script lets you plot a histogram for an estimation of the entropy of the video")
parser.add_argument('-gt','--groundTruthFile', action = 'store', dest = 'groundTruthFile', required = 'True', help = 'Ground Truth annotations')

arguments = parser.parse_args()

annotations = np.loadtxt(arguments.groundTruthFile,delimiter=",")
IDHash = {}

for annotationNumber,annotation in enumerate(annotations):
    idNumber = annotation[1]
    if idNumber in IDHash:
        IDHash[idNumber] += 1
    else:
        IDHash[idNumber] = 0

plt.hist(list(IDHash.values()),bins=50)
plt.xlabel('Number of frames a person stays')
plt.ylabel('Number of people')

plt.grid()
plt.show()
