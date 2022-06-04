import cv2
import os
from cv2 import contourArea
import numpy as np

def generate_mask(image, box, rect):
    #Logic for creating mask this doesnt work if rectangle tilt is further at position 4 than 3
    #corner diag
    # corner_pointx1, corner_pointy1 = box[0]
    # corner_pointx2, corner_pointy2 = box[2]
    # a = max(0, corner_pointy1)
    # b = max(0, corner_pointy2)
    # x1 = max(0, min(corner_pointx1, corner_pointx2))
    # x2 = max(0, max(corner_pointx1, corner_pointx2))
    # if (a > b):
    #     y1 = b
    #     y2 = a
    # else:
    #     y1 = a
    #     y2 = b
    # #mask = np.zeros_like(image)
    # #cv2.rectangle(mask, (x1,y1), (x2,y2), (255,255,255), -1)
    # masked_img = image[y1:y2, x1:x2]
    mult = 1  # I wanted to show an area slightly larger than my min rectangle set this to one if you don't
    W = rect[1][0]
    H = rect[1][1]

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    rotated = False
    angle = rect[2]

    if angle < -45:
        angle+=90
        rotated = True

    center = (int((x1+x2)/2), int((y1+y2)/2))
    size = (int(mult*(x2-x1)),int(mult*(y2-y1)))
    #cv2.circle(img_box, center, 10, (0,255,0), -1) #again this was mostly for debugging purposes

    M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)

    cropped = cv2.getRectSubPix(image, size, center)    
    cropped = cv2.warpAffine(cropped, M, size)

    croppedW = W if not rotated else H 
    croppedH = H if not rotated else W

    masked_img = cv2.getRectSubPix(cropped, (int(croppedW*mult), int(croppedH*mult)), (size[0]/2, size[1]/2))

    cv2.imshow('isolate', masked_img)
    cv2.waitKey(0)
    
def isolate_img(image, box, rect):
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
            [0, 0],
            [width-1, 0],
            [width-1, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped

#This file will extract tiles from uneven sprite sheet
image = cv2.imread('/home/cvdarbeloff/Documents/kuckles.png') #Load image
save_path = '/home/cvdarbeloff/Documents/gan_pictures' #save photos to this folder
img_counter = 0

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("before", image)
cv2.waitKey(0)
#print(image.shape)
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
            #generate_mask(image, box, rect)
            iso_img = isolate_img(image, box, rect)
            
            #Export isolated image to folder to store images
            #cv2.imshow('isolate', iso_img)
            #cv2.waitKey(0)
            #adjust name of each image we want to store
            img_name = "kuckles_GBA_SonicBattle_{}.jpg".format(img_counter)
            # store image in correct folder
            currnet_pic = cv2.imwrite(os.path.join(save_path, img_name), iso_img)
            print("{} written".format(img_name))
            img_counter = img_counter + 1
            #cv2.drawContours(image,[box],0,(0,255,0),2)
            

cv2.imshow("cur_img", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

