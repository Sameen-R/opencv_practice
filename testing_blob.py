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
IMAGE_NAME = 'from_robot.PNG'
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

lower = np.array([0, 0, 200])
upper = np.array([150, 255, 255])

# lower = np.array([mean_l_h, mean_l_s, mean_l_v])
# upper = np.array([mean_u_h, mean_u_s, mean_u_v])

mask = cv2.inRange(img, lower, upper)

kernel = np.ones((3, 3), np.uint8)
opening_img = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

points = cv2.findNonZero(mask)
if cv2.countNonZero(mask) > 0:
    avg = np.mean(points, axis=0)
else:
    avg = None

# displaying
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = False
params.filterByArea = True
params.minArea = 300
params.filterByCircularity = False
params.filterByInertia = False
params.filterByConvexity = False
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(opening_img)
sumX = 0
sumY = 0
count = 0
for kp in keypoints:
    count+=1
    sumX+=kp.pt[0]
    sumY+=kp.pt[1]

print(f'Blobs: {count}')

print(str(sumX/3) + " " + str(sumY/3))
means = (sumX/3, sumY/3)
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