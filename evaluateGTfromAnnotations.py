#!/usr/bin/python3
import argparse
import numpy as np

## Argument Parsing
parser = argparse.ArgumentParser(description = "This script reports the True Positives/False Positives over different distances (pixelDistances) using the ground truth.")
parser.add_argument('-gt',"--groundTruth", action='store', dest='groundTruthFile', required = True, help = 'Ground Truth for the original video.')
parser.add_argument('-wt',"--videoWidth",action='store',dest='videoWidth', default=640, help = 'If the width isn\'t 640, please specify. Annotations are scaled accordingly.')
parser.add_argument('-ht',"--videoHeight",action='store',dest='videoHeight', default=480, help = 'If the height isn\'t 480, please specify. Annotations are scaled accordingly.')

arguments = parser.parse_args()

annotations = np.loadtxt(arguments.groundTruthFile,delimiter=",")

annotations[:,[2,4]] *= (640.0/float(arguments.videoWidth))
annotations[:,[3,5]] *= (480.0/float(arguments.videoHeight))

detectedPeople = {}

#Constants
skipFrame = 0 #Can be made 1 for resampling the video to half the frequency
overlapThreshold = 0.3
distanceList = (260,270,280,290,300,310,320,480)

distanceBuckets = {}
for dist in distanceList:
    distanceBuckets[dist] = 0

detectionsFromFile = np.loadtxt(arguments.groundTruthFile,delimiter=",")

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

detectionsFromFile[:,[2,4]] *= (640.0/float(arguments.videoWidth))
detectionsFromFile[:,[3,5]] *= (480.0/float(arguments.videoHeight))

for detectionNumber in range(0,detectionsFromFile.shape[0]):
    frameNumber = detectionsFromFile[detectionNumber,0]

    if frameNumber % (skipFrame + 1) == 0:
        rect = detectionsFromFile[detectionNumber,2:6]
        #print("BBOX:",rect)
        overlap,personID = getIDForMaxOverlap(rect,annotations[annotations[:,0] == frameNumber])
        #print(personID)

        if overlap >= overlapThreshold:
            if personID in detectedPeople:
                detectedPeople[personID] += 1
            else:
                detectedPeople[personID] = 1
                yFeet = rect[1] + rect[3]
                #if yFeet > 480:
                #    print("Feet beyond 480", yFeet)
                for dist in distanceList:
                    if yFeet <= dist:
                        distanceBuckets[dist] += 1
        else:
            falsePositives += 1

    #print("in detection number : ",detectionNumber)

#IOU Unit test
#print(getIOU((2,2,1,1),(3,3,2,2)))

print("***************** SUMMARY **************")
print(len(detectedPeople), " people detected")
print("False Positives : ", falsePositives)

distanceList = np.array(distanceList)

#print(detectedPeople)
print("**************** True Positives *************")
print("pixelDistance, Cumulative Count (TP)")
for dist in distanceList:
    print(str(dist)+","+str(distanceBuckets[dist]))
