# import numpy as np
import cv2
import os
from sklearn.metrics import mean_squared_error

file_name = "VisaoTrab1.tiff"
file_name_no_extension = os.path.splitext(file_name)[0]

img = cv2.imread(file_name)
mean_squared_error(img, img)

for quality in [10, 50, 90]:
	cv2.imwrite(f'{file_name_no_extension}_{quality}.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

# Falta TIFF - LZW
