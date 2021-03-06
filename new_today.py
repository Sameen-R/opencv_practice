from cscore import CameraServer
from networktables import NetworkTables

import cv2
import json
import numpy as np
import time

blob_means = [(465.78973388671875, 305.0594075520833),
              (571.5816853841146, 245.081787109375),
              (449.79188028971356, 306.5421396891276),
              (516.1250610351562, 217.60052490234375)]

path_options = blob_means


def nothing(x):
    pass


def distance(x, y):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

def is_within_bounds(lx, hx, ly, hy, bgp):
    return (bgp.pt[0] < hx and bgp.pt[0] > lx) and (bgp.pt[1] < hy and bgp.pt[1] > ly)

def is_within_x(x, biggest_keypoints):
    if biggest_keypoints[-1].pt[0]<=x:
        return True
    else:
        return False

def main():
    width = 1280
    height = 720

    CameraServer.getInstance().startAutomaticCapture()

    input_stream = CameraServer.getInstance().getVideo()
    output_stream = CameraServer.getInstance().putVideo('Processed', width, height)
    blob_stream = CameraServer.getInstance().putVideo('Blobs', width, height)

    # Table for vision output information
    NetworkTables.initialize(server='10.25.2.2')

    # Wait for NetworkTables to start
    time.sleep(0.5)

    nt = NetworkTables.getTable('SmartDashboard')
    nt.putNumber('lower_h', 0)
    nt.putNumber('lower_s', 215)
    nt.putNumber('lower_v', 235)

    nt.putNumber('upper_h', 245)
    nt.putNumber('upper_s', 255)
    nt.putNumber('upper_v', 255)

    img = np.zeros(shape=(height, width, 3), dtype=np.uint8)

    while True:

        lower_h = nt.getNumber('lower_h', 0)
        lower_s = nt.getNumber('lower_s', 215)
        lower_v = nt.getNumber('lower_v', 235)

        upper_h = nt.getNumber('upper_h', 245)
        upper_s = nt.getNumber('upper_s', 255)
        upper_v = nt.getNumber('upper_v', 255)

        frame_time, img = input_stream.grabFrame(img)
        output_img = np.copy(img)

        # Notify output of error and skip iteration
        if frame_time == 0:
            # Send the output the error.
            output_stream.notifyError(input_stream.getError())
            # skip the rest of the current iteration
            continue

        # Convert to HSV and threshold image
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        binary_img = cv2.inRange(hsv_img, (65, 65, 200), (85, 255, 255))

        lower = np.array([lower_h, lower_s, lower_v])
        upper = np.array([upper_h, upper_s, upper_v])

        '''
        Try these lower/upper values
        lower = np.array([0, 215, 235])
        upper = np.array([245, 255, 255])
        '''

        mask = cv2.inRange(img, lower, upper)

        kernel = np.ones((3, 3), np.uint8)
        #opening_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        opening_img = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        opening_img = cv2.morphologyEx(opening_img, cv2.MORPH_OPEN, kernel)
        # opening_img = cv2.erode(opening_img, kernel, iterations = 1)

        points = cv2.findNonZero(mask)
        if cv2.countNonZero(mask) > 0:
            avg = np.mean(points, axis=0)
            #print(avg)
        else:
            avg = [0, 0]
        # displaying
        # params = cv2.SimpleBlobDetector_Params()
        # params.filterByColor = False
        # params.filterByArea = False
        # params.minArea = 5
        # params.filterByCircularity = False
        # params.filterByInertia = False
        # params.filterByConvexity = False
        # detector = cv2.SimpleBlobDetector_create(params)
        # keypoints = detector.detect(opening_img)
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = False
        params.filterByCircularity = False
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
        params.filterByInertia = False
        params.minInertiaRatio = 0.3
        params.maxInertiaRatio = 1

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(opening_img)

        print(len(keypoints))

        keypoints = sorted(keypoints, key=lambda keypoint: keypoint.size, reverse=True)

        biggest_keypoints = []

        if len(keypoints) == 1:
            nt.putNumberArray('big1', keypoints[0].pt)
            print(keypoints[0].pt)
            biggest_keypoints = [keypoints[0]]
        if len(keypoints) == 2:
            biggest_keypoints = keypoints[:2]
            nt.putNumberArray('big1', keypoints[0].pt)
            nt.putNumberArray('big2', keypoints[1].pt)
        if len(keypoints) >= 3:
            biggest_keypoints = keypoints[:3]
            nt.putNumberArray('big1', keypoints[0].pt)
            nt.putNumberArray('big2', keypoints[1].pt)
            nt.putNumberArray('big3', keypoints[2].pt)

        im_with_keypoints = cv2.drawKeypoints(opening_img, biggest_keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        '''
        (Color, Layout) 
        Color: 0->red, 1->blue
        Layout: 0->PathA, 1->PathB
        '''
        if len(biggest_keypoints) != 0:
            if is_within_bounds(113, 133, 34, 54, biggest_keypoints[0]):
                nt.putNumber('Path Number', 4)
            elif is_within_bounds(145, 165, 35, 55, biggest_keypoints[0]):
                nt.putNumber('Path Number', 2)
            elif is_within_bounds(83, 103, 49, 69, biggest_keypoints[0]):
                nt.putNumber('Path Number', 1)
            elif is_within_bounds(0, 20, 46, 66, biggest_keypoints[0]):
                nt.putNumber('Path Number', 3)
            else:
                nt.putNumber('Path Number', 0)
        else:
            nt.putNumber('Path Number', 0)

            

        #if is_within_y(960*.75, biggest_keypoints):
        #    nt.putNumber('Path #', 1)
        # if len(biggest_keypoints) > 0:
        #     nt.putNumber('Average X', (sum([kp.pt[0] for kp in biggest_keypoints])) / len(biggest_keypoints))
        #     nt.putNumber('Average Y', (sum([kp.pt[1] for kp in biggest_keypoints])) / len(biggest_keypoints))
        # else:
        #     nt.putNumber('Average X', 0)
        #     nt.putNumber('Average Y', 0)
        output_stream.putFrame(mask)
        blob_stream.putFrame(im_with_keypoints)


if __name__ == '__main__':
    main()
