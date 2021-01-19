import cv2
import time
import numpy as np
from scipy.spatial import distance
from means import mean_l_h, mean_l_s, mean_l_v, mean_u_h, mean_u_s, mean_u_v

'''
COMMENTED INSTRUCTIONS
In the imread() line, use the IMAGE_NAME variable
Then, run the program and press 'm' to predict the path
'''
#To find the predicted path, run the program and press 'm'
IMAGE_NAME = 'obj3.jpg'
path_options = [(479.89423077, 325.51442308), (619.56692913, 251.41994751), (408.97231834, 343.41176471), (523.62751678, 226.62080537)]

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
    img = cv2.imread(IMAGE_NAME, -1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # what to do with each vid frame
    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")
    h_h = cv2.getTrackbarPos("UH", "Tracking")
    h_s = cv2.getTrackbarPos("US", "Tracking")
    h_v = cv2.getTrackbarPos("UV", "Tracking")

    # lower = np.array([mean_l_h, mean_l_s, mean_l_v])
    # upper = np.array([mean_u_h, mean_u_s, mean_u_v])

    lower = np.array([l_h, l_s, l_v])
    upper = np.array([h_h, h_s, h_v])

    mask = cv2.inRange(img, lower, upper)

    kernel = np.ones((3,3), np.uint8)
    opening_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    points = cv2.findNonZero(mask)
    if cv2.countNonZero(mask)>0:
        avg = np.mean(points, axis=0)
    else:
        avg = None

    # displaying
    cv2.imshow('processed image', mask)
    k = cv2.waitKey(1)
    if k==ord('q'):
        break
    elif k==ord('p'):
        print(f'LH:{l_h} LS:{l_s} LV:{l_v} >>> UH:{h_h} US:{h_s} UV:{h_v}')
    elif k==ord('s'):
        print(f'shape: {mask.shape}')
    elif k==ord('m'):
        min_dist = -1
        index = -1
        for i in range(0,4):
            if distance.euclidean(path_options[i],avg)<min_dist or min_dist==-1:
                min_dist = distance.euclidean(path_options[i],avg)
                index = i
        print(f'Path # {index+1}')


cv2.destroyAllWindows()

'''
Obj1: mean=480.52324037 327.4873838  new mean: 479.89423077 325.51442308 
Lh:0
Ls:224
Lv:237
Uh:40
Us:255
Uv:255

Obj2: mean=615.16331658 250.83417085  new mean=619.56692913 251.41994751
Lh:0
Ls:230
Lv:245
Uh:105
Us:255
Uv:255

Obj3: mean=409.30617284 342.99012346  new mean=408.97231834 343.41176471
Lh:0
Ls:232
Lv:242
Uh:85
Us:255
Uv:255

Obj4: mean=525.56470588 224.36470588  new mean=523.62751678 226.62080537
Lh:0
Ls:225
Lv:240
Uh:82
Us:255
Uv:255
'''