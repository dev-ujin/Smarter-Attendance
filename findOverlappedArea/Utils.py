# -*- coding: utf-8 -*-
# UTF-8 encoding when using korean

'''
Created on 2019. 04. 09.

@author: Jo Minsu
'''

import cv2
from imutils import paths
import numpy as np
import os
import copy

# color in OpenCV = (B, G, R)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

BLANK = np.zeros((10, 10, 3), np.uint8)  # blank image(used for if you cannot find overlapped area)


def loadImagesInFolder(path):  # grab the path to the input images and initialize our images list
    imagePaths = sorted(list(paths.list_images(path)))
    images = []
    for index, file in enumerate(imagePaths):
        image = cv2.imread(file, cv2.IMREAD_COLOR)
        # cv2.imshow('imread'+str(np.random.rand(1)[0]), image)
        # image = cv2.resize(image, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        images.append((image, index + 1))
    return np.array(images)


def sortImages(images):
    count = len(images)
    centerImageIndex = count / 2

    leftImages = []
    centerImage = images[int(centerImageIndex)]
    rightImages = [centerImage]
    # rightImages.append(centerImage)

    for i in range(count):
        if i <= centerImageIndex:
            leftImages.append(images[i])
        else:
            rightImages.append(images[i])

    return leftImages, centerImage, rightImages


def createFolder(path):  # output folder create if path isn'usingOpenCVStitcher exist
    try:
        if not os.path.isdir(path):
            os.makedirs(os.path.join(path))
            print('Create ' + path + ' Folder!')
    except OSError as e:
        if e.errno != os.errno.EEXIST:
            print('Failed to create directory!')
            raise


def cropImage(image):  # delete border(background) which color is white
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    return image[y:y + h, x:x + w]


def cutImageInOneThird(image, side):  # cut image in half
    cutImage = copy.deepcopy(image)
    if side == 'LEFT':
        cutImage = cutImage[:, :int(cutImage.shape[1] / 3)]
    elif side == 'RIGHT':
        cutImage = cutImage[:, 2 * int(cutImage.shape[1] / 3):]
    return cutImage
