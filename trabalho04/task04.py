import cv2
import numpy as np
import matplotlib.pyplot as plt

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


print("Exemplo PDF")
xy = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
uv = np.array([[2, 1], [3, 5], [6, 6], [7, 2]])
print(minimos_quadrados(xy, uv))

print("Storm Trooper")
xy = np.array([[32, 1074], [526, 107], [1410, 226], [1248, 1311]])
uv = np.array([[264, 1142], [264, 264], [1076, 260], [1076, 1142]])
T = minimos_quadrados(xy, uv)
print(T)

file_name = "storm_trooper.jpg"
storm_img = cv2.imread(file_name, 0)
