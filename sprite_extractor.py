import cv2
from cv2 import contourArea
import numpy as np

#This file will extract tiles from uneven sprite sheet
image = cv2.imread('/home/cvdarbeloff/Documents/kuckles.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("before", image)
cv2.waitKey(0)
print(image.shape)
params = cv2.SimpleBlobDetector_Params()
ret,thresh = cv2.threshold(gray,127,255,0) # coverts grayscale to a binary img
contours,hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Set my contour size limit
area_limt = 300
for i, cnt in enumerate(contours):
        cur_area = cv2.contourArea(cnt)
        #print(i, len(cnt), cv2.contourArea(cnt))
        if cur_area > area_limt:
            #This means we found a good contour
            rect = cv2.minAreaRect(np.array(cnt))
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(image,[box],0,(0,255,0),2)
#cv2.drawContours(image, contours, -1, (0,255,0), 5)
#Change thresholds

# params.minThreshold = 50
# params.maxThreshold = 200

# params.filterByArea = True
# params.minArea = 10

# detector = cv2.SimpleBlobDetector_create(params)
# keypoints = detector.detect(image)

# blank = np.zeros((1, 1))
# blobs = cv2.drawKeypoints(image, keypoints, blank, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS )
# print(keypoints)
cv2.imshow("cur_img", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

