import numpy as np
import heapq
from numpy.__config__ import show
from numpy.core.fromnumeric import shape
import cv2 as cv
from matplotlib import pyplot as plt
from blend import *


def main():

    img1_color = cv.imread('./data/test1.jpg') # queryImage
    img2_color = cv.imread('./data/test2.jpg') # trainImage (Reference)

    img1 = cv.cvtColor(img1_color, cv.COLOR_BGR2GRAY) # queryImage
    img2 = cv.cvtColor(img2_color, cv.COLOR_BGR2GRAY) # trainImage (Reference)

    h,w = img1.shape

    # Initiate KAZE detector
    kaze = cv.KAZE_create()

    # find the keypoints and descriptors with KAZE
    # keyPoint object & float array
    kp1, des1 = kaze.detectAndCompute(img1,None)
    kp2, des2 = kaze.detectAndCompute(img2,None)

    # Set radius:
    radius         = 10
    matchThreshold = 10
    maxRatio       = 0.6
    confidence     = 99.9
    maxDistance    = 2
    maxNumTrials   = 2000

    # Convert images to grayscale
    matches = radiusMatch(des1, des2, kp2, kp1, radius, matchThreshold, maxRatio)

    #for match in matches:
    src_pts =  np.float32([ kp1[match[0]].pt for match in matches ]).reshape(-1,1,2)
    dst_pts =  np.float32([ kp2[match[1]].pt for match in matches ]).reshape(-1,1,2) 

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    
    transformed_img = cv.warpPerspective(img1_color,
                    M, (w, h))

    cv.imshow("i2", img2_color)
    #cv.imshow("test", helperBlendImages( transformed_img,img2_color ))
    cv.waitKey()
    
    cv.imshow("transformed", transformed_img)
    cv.waitKey()
    
    return 0


# Output:
# indexPairs    : Px2 matrix - indices of each pair of keypoints (in order of kp1, kp2)
# matchMetrics  : Px1 matrix - distance between each pair of match SAD (absolute difference) or SSD (squared difference)

def radiusMatch(feature1, feature2, point2, centerPoints, radius, matchThreshold, maxRatio):

    numCenter  = len(centerPoints)
    numPoints2 = len(point2)

    matchThreshold = matchThreshold*0.01*4

    # outputClass is np.float32 or anything 

    # Normalize features to unit vector

    for i in range(numCenter):
        feature1[i] = feature1[i] / np.linalg.norm(feature1[i])

    for i in range(numPoints2):
        feature2[i] = feature2[i] / np.linalg.norm(feature2[i])
    
    # Derive coordinates of each keypoints
    point1Location = np.zeros((numCenter, 2))
    point2Location = np.zeros((numPoints2, 2))

    for i in range(numCenter):
        point1Location[i][0] = centerPoints[i].pt[0]
        point1Location[i][1] = centerPoints[i].pt[1]
    

    for i in range(numPoints2):
        point2Location[i][0] = point2[i].pt[0]
        point2Location[i][1] = point2[i].pt[1]

    # Find all possible distances
    allDist = allSpatialDist(point1Location, point2Location)
    
    # Distances within radius
    inRadius = allDist <= radius**2
    
    # Find strong matches
    matchScores = np.zeros((numCenter, numPoints2)) # SSD(feature1, feature2)
    
    for i in range(numCenter):
        for j in range(numPoints2):
            matchScores[i][j] = SSD(feature1[i], feature2[j])

    # Boolean array numCenter x numPoints2
    isStrongMatch = matchScores <= matchThreshold
    
    trial = inRadius & isStrongMatch
    # print(matchScores[trial].shape)

    indexPairs = np.zeros((numCenter, 2)).astype(np.uint32)
    indicesPoint2 = np.arange(numPoints2)

    for centerIdx in range(numCenter):
        neighborIdxLogical = inRadius[centerIdx]
        candidateScores    = matchScores[centerIdx][neighborIdxLogical]
        neighborIdxLinear  = indicesPoint2[neighborIdxLogical]
        
        if len(neighborIdxLinear) == 1:
            matchIndex = neighborIdxLinear[0]

        elif len(neighborIdxLinear) > 1:

            topTwo = heapq.nsmallest(2,range(len(candidateScores)), key=candidateScores.__getitem__) # index of array candidateScores 
            topTwoIndices = np.array([neighborIdxLinear[topTwo[0]], neighborIdxLinear[topTwo[1]]])
            topTwoMetrics = np.array([candidateScores[topTwo[0]], candidateScores[topTwo[1]]]) # index of true keypoints

            matchIndex = topTwoIndices[0]

            # Ratio test
            if topTwoMetrics[1] < 1e-6:
                ratio = float(1)
            
            else:
                ratio = topTwoMetrics[0] / topTwoMetrics[1]
                #print(ratio)
            
            if ratio > maxRatio:
                continue  # Ambiguous match
        
        else:
            matchIndex = 0 # for codegen
            continue # no match found
        
        if isStrongMatch[centerIdx][matchIndex]:
            indexPairs[centerIdx][0] = centerIdx
            indexPairs[centerIdx][1] = matchIndex

   
    result = indexPairs[np.where((indexPairs[:,0]!= 0) & (indexPairs[:,1]!= 0))]

    # Check match uniqueness is implemented later since number of matches
    # are insufficient for homography matrix generation (>= 4 matches)

    return result




def SSD(point1, point2):
    return np.sum((point2 - point1)**2)


def allSpatialDist(points1, points2):
    numPoint1 = len(points1)
    numPoint2 = len(points2)
    allDist = np.zeros((numPoint1, numPoint2))

    for i in range(numPoint1):
        for j in range(numPoint2):
            allDist[i][j] = SSD(points1[i], points2[j])

    return allDist


if __name__ == "__main__":
    main()


