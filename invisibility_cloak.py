# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--video', help = """Path to input video file.
	Skip this argument to capture frames from a camera.""")

args = parser.parse_args()

print("""

	Hey !!! Would you like to try my invisibility cloak ???

	It's awesome !!!

	Prepare to get invisible ................. 
	""")

# img_path = 'img/circle.png'
# img = cv2.imread(img_path)
# cv2.imshow('img', img)

# mask = cv2.inRange(img, np.array([100, 100, 100]), np.array([255, 255, 255]))
# cv2.imshow('mask', mask)
# cv2.waitKey(0)
# print(mask + mask)

cap = cv2.VideoCapture(args.video if args.video else 0)

time.sleep(3)
count = 0
background = 0

for i in range(60):
	ret, background = cap.read()

background = np.flip(background, axis = 1)

while(cap.isOpened()):

	ret, img = cap.read()
	if not ret:
		break

	count += 1
	img = np.flip(img, axis = 1)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	lower_red = np.array([0, 120, 70])
	upper_red = np.array([10, 255, 255])
	mask1 = cv2.inRange(hsv, lower_red, upper_red)

	lower_red = np.array([170, 120, 70])
	upper_red = np.array([180, 255, 255])
	mask2 = cv2.inRnage(hsv, lower_red, upper_red)

	mask = mask1 + mask2

	# refining the mask corresponding to the detected red color
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations = 2)
	mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations = 1)
	mask2 = cv2.bitwise_not(mask)

	# generate the final output
	res1 = cv2.bitwise_and(background, background, mask = mask)
	res2 = cv2.bitwise_and(img, img, mask = mask2)
	final_output = cv2.addWeighted(res1, res2, 1, 0)

	cv2.imshow('Capture', final_output)
	k = cv2.waitKey(10)
	if k == 27:
		break 

		
