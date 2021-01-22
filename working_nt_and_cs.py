from cscore import CameraServer
from networktables import NetworkTables

import cv2
import json
import numpy as np
import time

mean_l_h = 0
LS = [224,230,232,225]
LV = [237,245,242,240]
UH = [40,105,85,82]
mean_u_s = 255
mean_u_v = 255

mean_l_s = np.mean(LS)
mean_l_v = np.mean(LV)
mean_u_h = np.mean(UH)

blob_means=[(465.78973388671875,305.0594075520833),
            (571.5816853841146,245.081787109375),
            (449.79188028971356,306.5421396891276),
            (516.1250610351562,217.60052490234375)]

path_options = blob_means

def nothing(x):
    pass


def distance(x, y):
    return np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)


def main():
   width = 960
   height = 544

   CameraServer.getInstance().startAutomaticCapture()

   input_stream = CameraServer.getInstance().getVideo()
   output_stream = CameraServer.getInstance().putVideo('Processed', width, height)

   # Table for vision output information
   NetworkTables.initialize(server='10.25.2.2')

   # Wait for NetworkTables to start
   time.sleep(0.5)
   
   vision_nt = NetworkTables.getTable('SmartDashboard')

   

   img = np.zeros(shape=(height,width,3), dtype=np.uint8)

   while True:
      frame_time, img = input_stream.grabFrame(img)
      output_img = np.copy(img)

      # Notify output of error and skip iteration
      if frame_time == 0:
          # Send the output the error.
          outputStream.notifyError(cvSink.getError());
          # skip the rest of the current iteration
          continue

      # Convert to HSV and threshold image
      hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
      binary_img = cv2.inRange(hsv_img, (65, 65, 200), (85, 255, 255))


      lower = np.array([0, 0, 200])
      upper = np.array([150, 255, 255])

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
          dist = distance(path_options[i], means)
          if dist < min_dist or min_dist == -1:
              min_dist = dist
              index = i
      print(f'Path # {index + 1}')
      print(min_dist)

      vision_nt.putNumber('Path', index + 1)
      
      output_stream.putFrame(mask)


if __name__ == '__main__':
      main()
