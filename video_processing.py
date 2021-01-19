# Import the camera server
from cscore import CameraServer
import numpy as np
from scipy.spatial import distance
from means import mean_l_h, mean_l_s, mean_l_v, mean_u_h, mean_u_s, mean_u_v, blob_means


path_options = blob_means
# Import OpenCV and NumPy
import cv2
import numpy as np

def nothing(x):
    pass

def main():
    cs = CameraServer.getInstance()
    cs.enableLogging()

    # Capture from the first USB Camera on the system
    camera = cs.startAutomaticCapture()
    camera.setResolution(320, 240)

    # Get a CvSink. This will capture images from the camera
    cvSink = cs.getVideo()

    # (optional) Setup a CvSource. This will send images back to the Dashboard
    outputStream = cs.putVideo("Name", 320, 240)

    # Allocating new images is very expensive, always try to preallocate
    img = np.zeros(shape=(240, 320, 3), dtype=np.uint8)

    while True:
        # Tell the CvSink to grab a frame from the camera and put it
        # in the source image.  If there is an error notify the output.
        time, img = cvSink.grabFrame(img)
        cv2.namedWindow("Tracking")
        cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
        cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
        cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)

        cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
        cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
        cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

        # reading image
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # what to do with each vid frame
        l_h = cv2.getTrackbarPos("LH", "Tracking")
        l_s = cv2.getTrackbarPos("LS", "Tracking")
        l_v = cv2.getTrackbarPos("LV", "Tracking")
        h_h = cv2.getTrackbarPos("UH", "Tracking")
        h_s = cv2.getTrackbarPos("US", "Tracking")
        h_v = cv2.getTrackbarPos("UV", "Tracking")

        lower = np.array([0, 235, 175])
        upper = np.array([97, 255, 255])

        # lower = np.array([l_h, l_s, l_v])
        # upper = np.array([h_h, h_s, h_v])

        mask = cv2.inRange(img, lower, upper)

        kernel = np.ones((3, 3), np.uint8)
        opening_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        points = cv2.findNonZero(mask)
        if cv2.countNonZero(mask) > 0:
            avg = np.mean(points, axis=0)
        else:
            avg = None

        # displaying
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = False
        params.filterByArea = True
        params.minArea = 10
        params.filterByCircularity = False
        params.filterByInertia = False
        params.filterByConvexity = False
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(opening_img)
        sumX = 0
        sumY = 0
        count = 0
        for kp in keypoints:
            count += 1
            sumX += kp.pt[0]
            sumY += kp.pt[1]

        print(f'Blobs: {count}')

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
        if time == 0:
            # Send the output the error.
            outputStream.notifyError(cvSink.getError());
            # skip the rest of the current iteration
            continue

        #
        # Insert your image processing logic here!
        #

        # (optional) send some image back to the dashboard
        outputStream.putFrame(img)