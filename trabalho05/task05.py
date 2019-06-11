import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

np.seterr(all='raise')


def image_side_by_side(imgs, titles):
	imgs = [np.round(img).astype("uint8") for img in imgs]
	plt.subplot(121), plt.imshow(imgs[0], cmap='gray'), plt.title(titles[0])
	plt.subplot(122), plt.imshow(imgs[1], cmap='gray'), plt.title(titles[1])
	plt.show()


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


def abe(img, neigh):
	return cv2.dilate(cv2.erode(img, neigh, iterations=1), neigh.transpose(), iterations=1)


def fec(img, neigh):
	return cv2.erode(cv2.dilate(img, neigh, iterations=1), neigh.transpose(), iterations=1)


def tophat_white(img, neigh):
	return img - abe(img, neigh)


def tophat_black(img, neigh):
	return img - fec(img, neigh)


def find_circulos(img):
	img = np.round(img).astype("uint8")
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	circulos = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.3, 10)
	return np.round(circulos[0, :]).astype("int") if circulos is not None else []


def draw_circulos(img, circulos, cor_circulo=(0, 255, 0), cor_centro=(0, 128, 255)):
	output = img.copy()
	print(f"Foram encontrados {len(circulos) if circulos is not None else 0} circulos")

	for (x, y, r) in circulos:
		cv2.circle(output, (x, y), r, cor_circulo, 4)
		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), cor_centro, -1)
	return output


def less_circulos(circulos, min_distance=10):
	resp = []
	for (x, y, r) in circulos:
		incluir = True
		for (x1, y1, r1) in resp:
			if math.hypot(x1 - x, y1 - y) < min_distance:
				incluir = False
				break
		if incluir:
			resp += [(x, y, r)]
	return resp


file_name = "Cosmos_original.jpg"
cosmos_img = cv2.imread(file_name)
x, y = [40, 70], [90, 90]
cosmos_img_cropped = cosmos_img[x[0]:cosmos_img.shape[0] - x[1], y[0]:cosmos_img.shape[1] - y[1]]
image_side_by_side([cosmos_img, cosmos_img_cropped], ["Original", "Cortada"])
cosmos_img = cosmos_img_cropped

tophat5 = tophat_white(cosmos_img, strel(5))
image_side_by_side([cosmos_img, tophat5], ["Original", "tophat5"])
tophat3_cosmos = tophat5 - cosmos_img
image_side_by_side([cosmos_img, tophat3_cosmos], ["Original", "tophat5 - original"])

bordas_laplacian = cv2.Laplacian(tophat3_cosmos, cv2.CV_64F)
image_side_by_side([cosmos_img, bordas_laplacian], ["Original", "bordas_laplacian"])
ret, thresh1 = cv2.threshold(bordas_laplacian, 127, 255, cv2.THRESH_BINARY)
image_side_by_side([cosmos_img, thresh1], ["Original", "thresh1"])

thresh1_circulos = draw_circulos(cosmos_img, find_circulos(thresh1))
image_side_by_side([thresh1, thresh1_circulos], ["thresh1", "thresh1_circulos"])

tophat3_circulos = find_circulos(tophat3_cosmos)
tophat3_cosmos_circulos = draw_circulos(cosmos_img, tophat3_circulos)
image_side_by_side([tophat3_cosmos, tophat3_cosmos_circulos], ["tophat3_cosmos", "tophat3_cosmos_circulos"])

tophat3_circulos_less = less_circulos(tophat3_circulos, 20)
tophat3_cosmos_circulos_less = draw_circulos(cosmos_img, tophat3_circulos_less)
image_side_by_side([tophat3_cosmos, tophat3_cosmos_circulos_less], ["tophat3_cosmos", "tophat3_cosmos_circulos_less"])
