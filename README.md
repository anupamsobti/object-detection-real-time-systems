This repository provides a framework for analyzing the performance of different detectors at arbitrary frame processing rates (FPRs). We have provided sample ground truth and detections for 5 videos from the MOT Challenge Dataset, namely, Bahnhof, MOT16-02, MOT16-05, MOT16-09, MOT16-10, MOT16-11. The detections are analyzed and classified as a True/False positive by the script and a segregation is done for different pixelDistances.

For more details, have a look at our paper [here](http://www.cse.iitd.ac.in/~anupam/obj-det-wacv18.pdf). It has been accepted at WACV '18.

___________________________________

## Dependencies
- Python3
- Numpy
- Matplotlib

___________________________________

A brief description of the files included is as follows:

File Name | Description | Usage
:--- | :-------------------------------- | :------------- 
analyseDetections.py | This script reports the True Positives/False Positives over different distances (pixelDistances) using detections from detectionResults/ and ground truth from gt/ . | Sample: `./analyseDetections.py -d detectionResults/bahnhof_ssd_mobilenet_v1_coco_11_06_out.txt -s 15 -gt gt/bahnhof_gt.txt`. Use `./analyseDetections.py -h` for sample usage. 
evaluateGTfromAnnotations.py | This script reports the True Positives/False Positives over different distances (pixelDistances) using the ground truth. | Use `./evaluateGTfromAnnotations.py -h` for sample usage. 
getIDDistribution.py | This script lets you plot a histogram for an estimation of the entropy of the video | Sample: `./getIDDistribution.py -gt gt/bahnhof_gt.txt`. Use `./getIDDistribution.py -h` for sample usage. 
