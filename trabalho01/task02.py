import numpy as np
import cv2
import os

file_name = "VisaoTrab1.tiff"
file_name_no_extension = os.path.splitext(file_name)[0]

img = cv2.imread(file_name)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

gamma = 1.5
channel_max = 179.0
channel_h = np.zeros(hsv.shape[:2])
for x in range(hsv.shape[0]):
	for y in range(hsv.shape[1]):
		channel_h[x, y] = hsv[x, y][0] / channel_max
channel_h = channel_h * gamma * channel_max

for x in range(hsv.shape[0]):
	for y in range(hsv.shape[1]):
		hsv[x, y][0] = int(max(min(channel_h[x, y], channel_max), 0))

img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite(f'{file_name_no_extension}_{gamma}.tiff', img)

# print("oi")
