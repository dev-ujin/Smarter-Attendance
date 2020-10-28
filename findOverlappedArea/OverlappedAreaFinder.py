# -*- coding: utf-8 -*-
# UTF-8 encoding when using korean

'''
Created on 2019. 03. 26.

@author: Jo Minsu
'''

import cv2
cv2.ocl.setUseOpenCL(False)
import numpy as np
from findOverlappedArea.Utils import cropImage, cutImageInOneThird, WHITE, BLACK, GREEN, BLANK
import copy

def getKpDesUsingORB(image):
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # change image's color-space
    resultImage = None

    orb = cv2.ORB_create() # orb object initialize
    keyPoints, descriptors = orb.detectAndCompute(grayImage, None) # calculate keyPoints and descriptors
    #print(type(keyPoints), type(descriptors))

    resultImage = cv2.drawKeypoints(image, keyPoints, resultImage, GREEN, flags=0) # draw keyPoints in image
    return keyPoints, descriptors, resultImage

def featureMatchingUsingBFORB(descriptors): # feature matching using BF
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors[0], descriptors[1])
    matches = sorted(matches, key=lambda x:x.distance)
    return matches

def findHomographyAndMatchesMask(keyPoints1, keyPoints2, matches):
    if len(matches) >= 4: # 4점 이상으로 homography를 찾는 것이 가능함
        # 두 이미지에서 일치하는 키포인트의 위치를 ​​추출합니다. 그들은 perpective 변형을 찾기 위해 전달됩니다.
        sourcePoints = np.float32([keyPoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        destinationPoints = np.float32([keyPoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Use the RANSAC algorithm to estimate a homography matrix using our matched feature vectors.
        # 이 3x3 변환 행렬을 얻으면 이를 사용하여 queryImage의 모서리를 trainImage의 해당 점으로 변환합니다.
        homography, mask = cv2.findHomography(sourcePoints, destinationPoints, cv2.RANSAC, 5.0)
        return homography, mask.ravel().tolist()
    else: # 그렇지 않으면 단순히 일치하는 내용이 충분하지 않다는 메시지를 표시하십시오.
        print("Not enough matches are found - %d/%d" % (len(matches), 4))
        return None

def drawOverlappedArea(image1, image2, homography):
    height, width, _ = image1.shape
    pts = np.float32([[0, 0], [0, height-1], [width-1, height-1], [width-1, 0] ]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, homography)
    return cv2.polylines(image2, [np.int32(dst)], True, (0, 0, 255), 5, cv2.LINE_AA)

def eraseOverlappedArea(image1, image2, homography):
    height, width, _ = image1.shape
    pts = np.float32([[0, 0], [0, height-1], [width-1, height-1], [width-1, 0] ]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, homography)
    return cv2.fillPoly(image2, [np.int32(dst)], BLACK, cv2.LINE_AA)

def CreateOverlappedAreaMask(imageShape, homography):
    # create white image
    mask = np.zeros(imageShape, np.uint8)
    mask[:] = WHITE

    # find overlapped area
    height, width, _ = mask.shape
    pts = np.float32([[0, 0], [0, height-1], [width-1, height-1], [width-1, 0] ]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, homography)

    # draw polygon for overlapped area
    mask = cv2.fillPoly(mask, [np.int32(dst)], BLACK, cv2.LINE_AA)
    return mask

def overlappedImageIsTrustworthy(overlappedImage, blankRatio=0.6): # 찾아낸 영역에서 흰 공간이 50% 이상이면 잘못찾았다고 판단
    allPixelCount = overlappedImage.shape[0]*overlappedImage.shape[1]
    whitePixelCount = np.sum(overlappedImage == WHITE)#;print('(', overlappedImage.shape[0], overlappedImage.shape[1], ')', whitePixelCount, allPixelCount, whitePixelCount/allPixelCount)
    if whitePixelCount/allPixelCount >= blankRatio:
        return False
    else:
        return True

def findOverlappedAreaInTwoImages(images, retry=0): # stitchRightSideTwoImages
    image2, image1 = images

    # find key-point and descriptor in 2-image
    keyPoints1, descriptor1, image1_keypointed = getKpDesUsingORB(image1)
    keyPoints2, descriptor2, image2_keypointed = getKpDesUsingORB(image2)
    cv2.imshow('keyPoints_1_'+str(np.random.rand(1)[0]), image1_keypointed)
    cv2.imshow('keyPoints_2_'+str(np.random.rand(1)[0]), image2_keypointed)

    # find good points
    matches = featureMatchingUsingBFORB((descriptor1, descriptor2))
    if retry == 1: # already keyPoints are smaller and incorrect
        matches = matches[:int(len(matches)/2)] # so we pick least 4 points
    elif retry == 2:
        matches = matches[:4]

    # find homography and mask
    M = findHomographyAndMatchesMask(keyPoints1, keyPoints2, matches)

    if M is not None:
        homography, mask = M

        # get feature matched image
        drawParameters = dict(matchColor = (0, 255, 0), singlePointColor = None, matchesMask = mask, flags = 2)
        featureMatchedImage = cv2.drawMatches(image1, keyPoints1, image2, keyPoints2, matches, None, **drawParameters)
        cv2.imshow('matched_'+str(np.random.rand(1)[0]), featureMatchedImage)

        # draw overlapped area in image2(left image)
        overlappedAreaDrawedImage = copy.deepcopy(image2)
        overlappedAreaDrawedImage = drawOverlappedArea(image1, overlappedAreaDrawedImage, homography)
        #cv2.imshow('overlapped area_draw_'+str(np.random.rand(1)[0]), overlappedAreaDrawedImage)

        # erase overlapped area in image2(left image)
        overlappedAreaErasedImage = copy.deepcopy(image2)
        overlappedAreaErasedImage = eraseOverlappedArea(image1, overlappedAreaErasedImage, homography)
        #cv2.imshow('overlapped area_erase_'+str(np.random.rand(1)[0]), overlappedAreaErasedImage)

        # extract overlapped image from image2
        overlappedAreaMask = CreateOverlappedAreaMask(image2.shape, homography)
        overlappedAreaExtractedImage = cv2.add(image2, overlappedAreaMask)
        overlappedAreaExtractedImage = cropImage(overlappedAreaExtractedImage)

        # if, overlapped area in two images is too small, we can retry
        if not(overlappedImageIsTrustworthy(overlappedAreaExtractedImage)) and (retry == 0):
            print(' ---- I think overlapped area is too little...')
            print('Retrying 1...', end='')
            newLeftImage = cutImageInOneThird(image2, 'RIGHT') # left image cut right-side
            newRightImage = cutImageInOneThird(image1, 'LEFT') # right image cut left-side
            return findOverlappedAreaInTwoImages((newLeftImage, newRightImage), retry=1)

        if not(overlappedImageIsTrustworthy(overlappedAreaExtractedImage)) and (retry == 1):
            print(' ---- I think overlapped area is too little...')
            print('Retrying 2...', end='')
            return findOverlappedAreaInTwoImages((image2, image1), retry=2)

        # if, overlapped area in two images isn't trustworthy, can't find overlapped area
        if not(overlappedImageIsTrustworthy(overlappedAreaExtractedImage)) and (retry == 2):
            print(" ---- I think overlapped area isn't trustworthy...")
            print('Please re-take pictures!', end='')
            overlappedAreaDrawedImage = BLANK
            overlappedAreaErasedImage = BLANK
            overlappedAreaExtractedImage = BLANK

        return overlappedAreaDrawedImage, overlappedAreaErasedImage, overlappedAreaExtractedImage
    else:
        print('Cannot calculate homography')
        return None

def findOverlappedAreaInAllImages(images): # find overlapped area in all images
    theNumberOfImages = len(images)
    overlappedAreaImages = []
    overlappedAreaErasedImages = []
    overlappedAreaDrawedImages = []
    for i in range(1, theNumberOfImages):
        print('Find overlapped area in image('+str(images[i-1, 1])+', '+str(images[i, 1])+')', end='')
        overlappedAreaInTwoImages = findOverlappedAreaInTwoImages([images[i-1, 0], images[i, 0]])
        if overlappedAreaInTwoImages is not None:
            overlappedAreaDrawedImage, overlappedAreaErasedImage, overlappedAreaImage = overlappedAreaInTwoImages
            overlappedAreaImages.append(overlappedAreaImage)
            overlappedAreaErasedImages.append(overlappedAreaErasedImage)
            overlappedAreaDrawedImages.append(overlappedAreaDrawedImage)
            print(' ---- {0:.2f}%'.format((i/(theNumberOfImages-1))*100))
        else:
            print()
            print('Cannot find overlapped area in images')
            return None # 실패 이미지로 끼워넣는게 좋을듯...
    print('Complete find overlapped area in all images!')
    return overlappedAreaImages, overlappedAreaDrawedImages