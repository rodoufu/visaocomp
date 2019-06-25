import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def image_side_by_side(imgs, titles):
	imgs = [np.round(img).astype("uint8") for img in imgs]
	plt.subplot(121), plt.imshow(imgs[0], cmap='gray'), plt.title(titles[0])
	plt.subplot(122), plt.imshow(imgs[1], cmap='gray'), plt.title(titles[1])
	plt.show()


img = cv.imread('TI_0001.tif')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
ret, thresh = cv.threshold(gray, 50, 100, cv.THRESH_OTSU)

image_side_by_side([gray, thresh], ['Original', 'Threshold'])

# noise removal
kernel = np.ones((5, 5), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=3)
image_side_by_side([thresh, opening], ['Original', 'Noise removal'])

colorir = np.array(img, copy=True)

print(f"Valores encontrados: {np.unique(opening)}")

for x in range(colorir.shape[0]):
	for y in range(colorir.shape[1]):
		# print(f"{x},{y}={opening[x, y]}")
		if opening[x, y] >= 100:
			colorir[x, y] = np.array([0, 255, 0])


image_side_by_side([img, colorir], ['Original', 'Segmentado'])
