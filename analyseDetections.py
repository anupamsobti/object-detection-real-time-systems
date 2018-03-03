#!/usr/bin/python3

import argparse
import numpy as np

## Argument Parsing
parser = argparse.ArgumentParser(description = "This script reports the True Positives/False Positives over different distances (pixelDistances) using detections from detectionResults/ and ground truth from gt/ .")
parser.add_argument('-d',"--detectionsFile" , action='store',dest='detectionsFile',required=True, help='Detections from the detector/video combination to be processed')
parser.add_argument('-s',"--fpr", action='store',dest='detectorSpeed', required=True, help = 'fpr = Frame Processing Rate. This is the speed at which the detector is running. This determines how many frames are skipped.')
parser.add_argument('-gt',"--groundTruth", action='store', dest='groundTruthFile', required = True, help = 'Ground Truth for the original video.')
parser.add_argument('-wt',"--videoWidth",action='store',dest='videoWidth', default=640, help = 'If the width isn\'t 640, please specify. Annotations are scaled accordingly.')
parser.add_argument('-ht',"--videoHeight",action='store',dest='videoHeight', default=480, help = 'If the height isn\'t 480, please specify. Annotations are scaled accordingly.')

arguments = parser.parse_args()

#Constants
videoFPS = 15   #This is the frame rate at which the video is recorded
timePerFrame = 1.0/videoFPS

annotations = np.loadtxt(arguments.groundTruthFile,delimiter=",")

annotations[:,[2,4]] *= (640.0/float(arguments.videoWidth))
annotations[:,[3,5]] *= (480.0/float(arguments.videoHeight))

totalFrames = np.max(annotations[:,0])

detectedPeople = {}

#Constants
overlapThreshold = 0.3
distanceList = (260,270,280,290,300,310,320,480)

distanceBuckets = {}
falsePositiveBuckets = {}
for dist in distanceList:
    distanceBuckets[dist] = 0
    falsePositiveBuckets[dist] = 0

detectionsFromFile = np.loadtxt(arguments.detectionsFile,delimiter=",")
detectorSpeed = float(arguments.detectorSpeed)
timePerDetection = 1/detectorSpeed

def getIOU(rect1,rect2):
    (x1,y1,w1,h1) = rect1
    (x3,y3,w2,h2) = rect2

    (x2,y2,x4,y4) = (x1+w1,y1+h1,x3+w2,y3+h2)
    intersection = (max(x1,x3),max(y1,y3),min(x2,x4),min(y2,y4))

    if intersection[0] >= intersection[2] or intersection[1] >= intersection[3]:
        intersectArea = 0
        return 0.0
    else:
        intersectArea = (intersection[2] - intersection[0]) * (intersection[3] - intersection[1])

    unionArea = w1*h1 + w2*h2 - intersectArea

    return intersectArea/unionArea

def getIDForMaxOverlap(rect,annotation):
    """ Receives annotations for only the current frameNumber """
    #print(annotation.shape)
    #print(annotation)
    rectanglesForFrame = annotation[:,2:6]
    #print(rectanglesForFrame)
    #print(rectanglesForFrame.shape)
    ious = [getIOU(rect,x) for x in rectanglesForFrame]
    maxRectIndex = np.argmax(ious)
    #print(ious[maxRectIndex])
    return ious[maxRectIndex],annotation[maxRectIndex,1]


falsePositives = 0

frameNumber = 1
iterationNumber = 1
while frameNumber <= totalFrames:
    frameNumber = int(iterationNumber * timePerDetection/timePerFrame)
    iterationNumber += 1
    #print("Processing Frame : ", frameNumber)
    detectionsForCurrentFrame = detectionsFromFile[detectionsFromFile[:,0] == frameNumber]
    for rect in detectionsForCurrentFrame[:,1:5]:
        overlap,personID = getIDForMaxOverlap(rect,annotations[annotations[:,0] == frameNumber])
        if overlap >= overlapThreshold:
            if personID in detectedPeople:
                detectedPeople[personID] += 1
            else:
                detectedPeople[personID] = 1
                yFeet = rect[1] + rect[3]
                for dist in distanceList:
                    if yFeet <= dist:
                        distanceBuckets[dist] += 1
        else:
            yFeet = rect[1] + rect[3]
            for dist in distanceList:
                if yFeet <= dist:
                    falsePositiveBuckets[dist] += 1
            falsePositives += 1

#IOU Unit test
#print(getIOU((2,2,1,1),(3,3,2,2)))

print("***************** SUMMARY **************")
print(len(detectedPeople), " people detected")
print("False Positives : ", falsePositives)

distanceList = np.array(distanceList)

print("**************** True Positives *************")
print("pixelDistance, Cumulative Count (TP)")
for dist in distanceList:
    #print("FOR_TP_CSV: ",str(arguments.detectionsFile)+","+str(arguments.detectorSpeed)+","+str(dist)+","+str(distanceBuckets[dist]))
    print(str(dist)+","+str(distanceBuckets[dist]))

print()
print("**************** False Positives *************")
print("pixelDistance, False Positives")
for i,dist in enumerate(distanceList):
    if i >= 1:
        print(str(dist)+","+str(falsePositiveBuckets[dist] - falsePositiveBuckets[distanceList[i-1]]))
    else:
        print(str(dist)+","+str(falsePositiveBuckets[dist]))

print(iterationNumber, " frames processed of ", totalFrames)
