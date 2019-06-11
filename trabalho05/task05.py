import cv2
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')


def strel(radius):
	'''
	Cria um elemento estruturante em forma de círculo.
	:param radius: Raio do elemento estruturante.
	:return: O elemento estruturante.
	'''
	resp = np.zeros([2 * radius + 1, 2 * radius + 1], np.uint8)
	for i in range(2 * radius + 1):
		for j in range(2 * radius + 1):
			x = i - radius
			y = j - radius
			resp[i, j] = x * x + y * y <= radius * radius
	return resp


def show_image(img, title=None):
	if isinstance(img, list) and len(img) > 0:
		if isinstance(img[0], list) and len(img[0]) > 0:
			f, axarr = plt.subplots(len(img), len(img[0]), figsize=(16, 10))
			if title:
				f.suptitle(title, fontsize="x-large")
			for i in range(len(img)):
				for j in range(len(img[i])):
					axarr[i, j].imshow(img[i][j], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
		else:
			f, axarr = plt.subplots(1, len(img), figsize=(20, 10))
			if title:
				f.suptitle(title, fontsize="x-large")
			for i in range(len(img)):
				axarr[i].imshow(img[i], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
	else:
		if title:
			plt.title(title)
		plt.imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
	plt.show()


def abe(img, neigh):
	return cv2.dilate(cv2.erode(img, neigh, iterations=1), neigh.transpose(), iterations=1)


def fec(img, neigh):
	return cv2.erode(cv2.dilate(img, neigh, iterations=1), neigh.transpose(), iterations=1)


def tophat_white(img, neigh):
	return img - abe(img, neigh)


def tophat_black(img, neigh):
	return img - fec(img, neigh)


file_name = "Cosmos_original.jpg"
cosmos_img = cv2.imread(file_name)

tophat3 = tophat_white(cosmos_img, strel(5))
tophat3_cosmos = tophat3 - cosmos_img

bordas_laplacian = cv2.Laplacian(tophat3_cosmos, cv2.CV_64F)
show_image([tophat3_cosmos, bordas_laplacian])
ret, thresh1 = cv2.threshold(bordas_laplacian, 127, 255, cv2.THRESH_BINARY)

show_image([tophat3_cosmos, thresh1])
# gray = cv2.cvtColor(thresh1, cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(cosmos_img, cv2.COLOR_BGR2GRAY)
circulos = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.8, 200)

output = cosmos_img.copy()
# ensure at least some circles were found
if circulos is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circulos = np.round(circulos[0, :]).astype("int")

	# loop over the (x, y) coordinates and radius of the circles
	for (x, y, r) in circulos:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

	# show the output image
	# cv2.imshow("output", np.hstack([cosmos_img, output]))

show_image([cosmos_img, tophat3_cosmos, gray, output])