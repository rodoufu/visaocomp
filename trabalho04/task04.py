import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from enum import Enum

np.seterr(all='raise')


def minimos_quadrados(xy, uv):
	A = np.zeros([2 * xy.shape[0], 8])
	L = np.zeros([2 * xy.shape[0], 1])

	for i in range(xy.shape[0]):
		A[2 * i, 0], A[2 * i, 1], A[2 * i, 2] = xy[i][0], xy[i][1], 1
		A[2 * i, 6], A[2 * i, 7] = -xy[i][0] * uv[i][0], - xy[i][1] * uv[i][0]

		A[2 * i + 1, 3], A[2 * i + 1, 4], A[2 * i + 1, 5] = xy[i][0], xy[i][1], 1
		A[2 * i + 1, 6], A[2 * i + 1, 7] = -xy[i][0] * uv[i][1], - xy[i][1] * uv[i][1]

		L[2 * i], L[2 * i + 1] = uv[i, 0], uv[i, 1]

	A_t = A.transpose()
	A_tA_inv = np.linalg.inv(np.matmul(A_t, A))
	X_chapeu = np.matmul(np.matmul(A_tA_inv, A_t), L)

	T = np.ones([3, 3])
	T[0, 0], T[0, 1], T[0, 2] = X_chapeu[0], X_chapeu[1], X_chapeu[2]
	T[1, 0], T[1, 1], T[1, 2] = X_chapeu[3], X_chapeu[4], X_chapeu[5]
	T[2, 0], T[2, 1] = X_chapeu[6], X_chapeu[7]
	return T


file_name = "storm_trooper.jpg"
storm_img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
height, width = storm_img.shape

print("Storm Trooper")
xy = np.array([[107, 526], [226, 1410], [1311, 1248], [1074, 32]])
uv = np.array([(0, 0), (0, width), (width, height), (height, 0)])
transformation = minimos_quadrados(xy, uv)
print(transformation)


class PointMethod(Enum):
	NEAREST_NEIGHBOR = 1
	NEIGHBORHOOD_AVG = 2


def transformar_coordenadas(img, t, method=PointMethod.NEAREST_NEIGHBOR):
	resp = np.zeros((height, width), np.uint8)
	t = np.linalg.inv(t)

	for x in range(height - 1):
		for y in range(width - 1):
			x_chapeu = np.dot(t, np.array([x, y, 1]))
			x_chapeu /= x_chapeu[2]

			xv = [math.floor(x_chapeu[0]), math.ceil(x_chapeu[0])]
			yv = [math.floor(x_chapeu[1]), math.floor(x_chapeu[1])]
			points = [(i, j) for i in xv for j in yv]
			dists = [math.hypot(x_chapeu[0] - x_chapeu[1], it[0] - it[1]) for it in points]

			if method == PointMethod.NEAREST_NEIGHBOR:
				idx = np.argmin(dists)
				resp[x, y] = img[points[idx][0], points[idx][1]]
			elif method == PointMethod.NEIGHBORHOOD_AVG:
				value = 0
				for i in range(len(points)):
					value += img[points[i][0], points[i][1]] * dists[i]
				resp[x, y] = np.uint8(value // sum(dists))

	return resp


plt.imshow(storm_img, cmap='gray', vmin=0, vmax=255)
plt.show()
print(storm_img.shape)


def image_side_by_side(imgs, titles):
	plt.subplot(121), plt.imshow(imgs[0], cmap='gray'), plt.title(titles[0])
	plt.subplot(122), plt.imshow(imgs[1], cmap='gray'), plt.title(titles[1])
	plt.show()


start = time.time()
resp = transformar_coordenadas(storm_img, transformation, PointMethod.NEAREST_NEIGHBOR)
end = time.time()
print(f"Elapsed time: {end - start:3}s, size: {resp.shape}")
image_side_by_side([storm_img, resp], ['Input', 'NEAREST_NEIGHBOR'])

start = time.time()
resp_avg = transformar_coordenadas(storm_img, transformation, PointMethod.NEIGHBORHOOD_AVG)
end = time.time()
print(f"Elapsed time: {end - start:3}s, size: {resp_avg.shape}")
image_side_by_side([storm_img, resp_avg], ['Input', 'NEIGHBORHOOD_AVG'])

image_side_by_side([resp, resp_avg], ['NEAREST_NEIGHBOR', 'NEIGHBORHOOD_AVG'])

