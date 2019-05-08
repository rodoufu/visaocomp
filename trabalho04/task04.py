import cv2
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, njit, prange
import time

np.seterr(all='raise')


@jit(parallel=True)
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


# print("Exemplo PDF")
# xy = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
# uv = np.array([[2, 1], [3, 5], [6, 6], [7, 2]])
# print(minimos_quadrados(xy, uv))

print("Storm Trooper")
xy = np.array([[32, 1074], [526, 107], [1410, 226], [1248, 1311]])
uv = np.array([[264, 1142], [264, 264], [1076, 260], [1076, 1142]])
transformation = minimos_quadrados(xy, uv)
print(transformation)

tr = cv2.getPerspectiveTransform(np.float32(xy), np.float32(uv))
print(tr)

# from scipy import interpolate
# x = np.arange(-5.01, 5.01, 0.25)
# y = np.arange(-5.01, 5.01, 0.25)
# xx, yy = np.meshgrid(x, y)
# z = np.sin(xx**2+yy**2)
# f = interpolate.interp2d(x, y, z, kind='cubic')
# xnew = np.arange(-5.01, 5.01, 1e-2)
# ynew = np.arange(-5.01, 5.01, 1e-2)
# znew = f(xnew, ynew)
# plt.plot(x, z[0, :], 'ro-', xnew, znew[0, :], 'b-')
# plt.show()


# @jit
def transformar_coordenadas(img, t):
	new_image_map = {}
	minx, miny = img.shape[0], img.shape[1]
	maxx, maxy = 0, 0
	# t = np.linalg.inv(t)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			xy = np.array([i, j, 1])
			uv = np.matmul(t, xy)
			uv = uv / uv[2]
			uv[0], uv[1] = int(uv[0] + .5), int(uv[1] + .5)
			minx = min(minx, uv[0])
			maxx = max(maxx, uv[0])
			miny = min(miny, uv[1])
			maxy = max(maxy, uv[1])

			# uv[0] = (t[0, 0] * i + t[0, 1] * j + t[0, 2]) / (t[2, 0] * i + t[2, 1] * j + t[2, 2])
			# uv[1] = (t[1, 0] * i + t[1, 1] * j + t[1, 2]) / (t[2, 0] * i + t[2, 1] * j + t[2, 2])

			new_image_map[int(uv[0]), int(uv[1])] = (i, j)

	minx, miny = int(minx), int(miny)
	maxx, maxy = int(maxx), int(maxy)
	final_img = np.zeros((maxx - minx + 1, maxy - miny + 1)) \
		if len(img.shape) == 2 else np.zeros((maxx - minx + 1, maxy - miny + 1, img.shape[2]))
	# final_img = np.zeros(img.shape)

	for k, v in new_image_map.items():
		final_img[k[0] - minx, k[1] - miny] = img[v]
		# if 0 <= k[0] < img.shape[0] and 0 <= k[1] < img.shape[1]:
		# 	final_img[k] = img[v]

	return final_img


file_name = "storm_trooper.jpg"
storm_img = cv2.imread(file_name)
plt.imshow(storm_img, vmin=0, vmax=255)
plt.show()
print(storm_img.shape)

dst = cv2.warpPerspective(storm_img, tr, (1448, 1456))
plt.subplot(121), plt.imshow(storm_img), plt.title('Input')
plt.subplot(122), plt.imshow(dst), plt.title('Output')
plt.show()

start = time.time()
resp = transformar_coordenadas(storm_img, transformation)
end = time.time()
print(f"Elapsed time: {end - start:3}s, size: {resp.shape}")
plt.imshow(resp, vmin=0, vmax=255)
plt.show()

# start = time.time()
# resp = transformar_coordenadas(storm_img, np.linalg.inv(transformation))
# end = time.time()
# print(f"Elapsed time: {end - start:3}s, size: {resp.shape}")
# plt.imshow(resp, vmin=0, vmax=255)
# plt.show()
