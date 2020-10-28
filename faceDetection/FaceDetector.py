#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from faceDetection.mtcnn import MTCNN
import matplotlib.pyplot as plt

def facedetector(inputpath, inputimage) :
    detector = MTCNN()
    image = cv2.imread(inputpath + inputimage)
    #image = cv2.imread("2.jpg")
    # 위에 코드에 읽을 사진 파일 이름 ex) 'sujin.jpg'
    print('start detecting faces!')
    result = detector.detect_faces(image)
    count=0
    for i in range(100): # 100명 이상 있는 교실은 없으니까 max 100
        try: # 100명 다 못채울 수 있으니까 try-except
            bounding_box = result[i]['box']
            keypoints = result[i]['keypoints']
            # Draw bounding box
            cv2.rectangle(image,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (0,155,255),
                  8)
            count += 1
        except:
            pass
    print('end detecting faces!')
    # circle -> 왼오 눈, 코,  왼오입
    cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)

    #cv2.imwrite("./detection_output/result.jpg", image)
    # cv2.imwrite()  첫번째로 복사한 이미지의 저장경로를 받고
    # 두번째는 복사할 이미지 객체를 파라미터로 받음
    # 코드를 실행하면 해당 경로에 복사된 이미지가 생성된 후 종료
    #plt.imshow(plt.imread("./detection_output/result.jpg"))
    #plt.show()
    return image, count