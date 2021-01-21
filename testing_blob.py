import cv2
import time
import numpy as np
from scipy.spatial import distance
from means import mean_l_h, mean_l_s, mean_l_v, mean_u_h, mean_u_s, mean_u_v, blob_means

'''
COMMENTED INSTRUCTIONS
In the imread() line, use the IMAGE_NAME variable
Then, run the program and press 'm' to predict the path
'''
#To find the predicted path, run the program and press 'm'
IMAGE_NAME = 'obj4.jpg'
# path_options = [(479.89423077, 325.51442308), (619.56692913, 251.41994751), (408.97231834, 343.41176471), (523.62751678, 226.62080537)]
path_options = blob_means

def nothing(x):
    pass

cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)

cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

while True:
    # reading image
    img = cv2.imread(IMAGE_NAME, 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # what to do with each vid frame
    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")
    h_h = cv2.getTrackbarPos("UH", "Tracking")
    h_s = cv2.getTrackbarPos("US", "Tracking")
    h_v = cv2.getTrackbarPos("UV", "Tracking")

    # lower = np.array([0, 0, 200])
    # upper = np.array([150, 255, 255])

    # lower = np.array([mean_l_h, mean_l_s, mean_l_v])
    # upper = np.array([mean_u_h, mean_u_s, mean_u_v])

    lower = np.array([0, 215, 235])
    upper = np.array([245, 255, 255])

    mask = cv2.inRange(img, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    opening_img = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opening_img = cv2.morphologyEx(opening_img, cv2.MORPH_OPEN, kernel)

    points = cv2.findNonZero(mask)
    if cv2.countNonZero(mask) > 0:
        avg = np.mean(points, axis=0)
    else:
        avg = None

    # displaying
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = False
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.maxCircularity = 1
    params.filterByConvexity = False

    '''
    Area needs to be between 100-500 if detecting the blue paths
    '''
    params.filterByArea = False
    # params.minArea = 100
    # params.maxArea = 1000

    '''
    Inertia ratio between 0.3-1 if detecting blue paths
    '''
    params.filterByInertia = True
    params.minInertiaRatio = 0.3
    params.maxInertiaRatio = 1

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(opening_img)

    big1 = None
    big2 = None
    big3 = None
    print(keypoints)

    if keypoints[0].size > keypoints[1].size:
        if keypoints[1].size > keypoints[2].size:
            big1 = keypoints[0]
            big2 = keypoints[1]
            big3 = keypoints[2]
        else:
            big1 = keypoints[0]
            big2 = keypoints[2]
            big3 = keypoints[1]

    else:
        if keypoints[0].size > keypoints[2].size:
            big1 = keypoints[1]
            big2 = keypoints[0]
            big3 = keypoints[2]
        else:
            big1 = keypoints[1]
            big2 = keypoints[2]
            big3 = keypoints[0]



    for kp in keypoints:
        if kp == big1:
            continue
        if kp == big2:
            continue
        if kp == big3:
            continue
        if kp.size > big1.size:
            big3 = big2
            big2 = big1
            big1 = kp
        elif kp.size > big2.size:
            big3 = big2
            big2 = kp
        elif kp.size > big3.size:
            big3 = kp

    keypoints = [big1, big2, big3]
    print(big1.pt)
    print(big2.pt)
    print(big3.pt)

    # far_points = []
    # for point in keypoints:
    #     if point.pt[1]<=320 and point.pt[1]>=170:
    #         if point.pt[0]<=780 and point.pt[0]>=190:
    #             far_points.append(point)


    sumX = 0
    sumY = 0
    count = 0


    print(str(sumX / 3) + " " + str(sumY / 3))
    means = (sumX / 3, sumY / 3)
    im_with_keypoints = cv2.drawKeypoints(opening_img, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    min_dist = -1
    index = -1
    for i in range(0, 4):
        if distance.euclidean(path_options[i], means) < min_dist or min_dist == -1:
            min_dist = distance.euclidean(path_options[i], means)
            index = i
    print(f'Path # {index + 1}')
    print(min_dist)
    cv2.imshow('processed image', im_with_keypoints)
    k = cv2.waitKey(0)
    if k == ord('p'):
        print(f'LH:{l_h} LS:{l_s} LV:{l_v} >>> UH:{h_h} US:{h_s} UV:{h_v}')
    elif k == ord('s'):
        print(f'shape: {mask.shape}')
    elif k == ord('m'):
        min_dist = -1
        index = -1
        for i in range(0, 4):
            if distance.euclidean(path_options[i], avg) < min_dist or min_dist == -1:
                min_dist = distance.euclidean(path_options[i], avg)
                index = i
        print(f'Path # {index + 1}')

    break


cv2.destroyAllWindows()

'''
Obj1: mean=480.52324037 327.4873838  new mean: 479.89423077 325.51442308 blob mean=465.78973388671875 305.0594075520833
Lh:0
Ls:224
Lv:237
Uh:40
Us:255
Uv:255

Obj2: mean=615.16331658 250.83417085  new mean=619.56692913 251.41994751 blob mean=571.5816853841146 245.081787109375
Lh:0
Ls:230
Lv:245
Uh:105
Us:255
Uv:255

Obj3: mean=409.30617284 342.99012346  new mean=408.97231834 343.41176471 blob mean=449.79188028971356 306.5421396891276
Lh:0
Ls:232
Lv:242
Uh:85
Us:255
Uv:255

Obj4: mean=525.56470588 224.36470588  new mean=523.62751678 226.62080537 blob mean=516.1250610351562 217.60052490234375
Lh:0
Ls:225
Lv:240
Uh:82
Us:255
Uv:255
'''