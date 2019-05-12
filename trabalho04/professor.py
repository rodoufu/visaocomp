import numpy as np

A = np.zeros([8, 8])
L = np.zeros([8, 1])
A[0, :] = [1074, 32, 1, 0, 0, 0, -1074 * 1142, -32 * 1142]
A[1, :] = [0, 0, 0, 1074, 32, 1, -1074 * 264, -32 * 264]
A[2, :] = [107, 526, 1, 0, 0, 0, - 107 * 264, - 526 * 264]
A[3, :] = [0, 0, 0, 107, 526, 1, - 107 * 264, - 526 * 264]
A[4, :] = [226, 1410, 1, 0, 0, 0, - 226 * 264, - 1410 * 264]
A[5, :] = [0, 0, 0, 226, 1410, 1, - 226 * 1076, - 1410 * 1076]
A[6, :] = [1311, 1248, 1, 0, 0, 0, -1311 * 1142, -1248 * 1142]
A[7, :] = [0, 0, 0, 1311, 1248, 1, -1311 * 1076, - 1248 * 1076]

L = [[1142], [264], [264], [264], [264], [1076], [1142], [1076]]
print(np.uint32(A))
print(np.uint32(L))

A_t = A.transpose()
A_tA_inv = np.linalg.inv(np.matmul(A_t, A))
X_chapeu = np.matmul(np.matmul(A_tA_inv, A_t), L)

T = np.ones([3, 3])
T[0, 0], T[0, 1], T[0, 2] = X_chapeu[0], X_chapeu[1], X_chapeu[2]
T[1, 0], T[1, 1], T[1, 2] = X_chapeu[3], X_chapeu[4], X_chapeu[5]
T[2, 0], T[2, 1] = X_chapeu[6], X_chapeu[7]

print(T)
P1 = np.array([[1074], [32], [1]])
P1_linha = np.matmul(T, P1)
P1_linha = P1_linha / P1_linha[2, 0]
print("P1")
print(np.uint32(P1))
print("P1_linha")
print(np.uint32(P1_linha))

P3 = np.array([[226], [1410], [1]])
P3_linha = np.matmul(T, P3)
P3_linha = P3_linha / P3_linha[2, 0]
print("P3")
print(np.uint32(P3))
print("P3_linha")
print(np.uint32(P3_linha))

T_inv = np.linalg.inv(T)
print("T_inv")
print(T_inv)
P1 = np.matmul(T_inv, P1_linha)
P1 = P1 / P1[2, 0]

print("P1_linha")
print(np.uint32(P1_linha))
print("P1")
print(np.uint32(P1))
