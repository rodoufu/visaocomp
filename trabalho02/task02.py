import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

image = np.array([
	[0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
	[0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
	[0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
	[0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
	[0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
	[0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
	[0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
	[0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
	[0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
	[0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
	[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


def passa_baixa(size):
	return np.array([[1.0/(size ** 2)] * size] * size)


def operacao_vizinhanca(img, neigh, min_value=0, max_value=255):
	resp = np.zeros(img.shape[:2])
	Imx = img.shape[0]
	Imy = img.shape[1]
	Nx = neigh.shape[0]
	Ny = neigh.shape[1]

	# Ignorando as bordas
	for x in range(Nx // 2):
		for y in range(Imy):
			resp[x, y] = img[x, y]
			resp[Imx - x - 1, y] = img[Imx - x - 1, y]
	for x in range(Imx):
		for y in range(Ny // 2):
			resp[x, y] = img[x, y]
			resp[x, Imy - y - 1] = img[x, Imy - y - 1]

	for x in range(Nx // 2, img.shape[0] - Nx // 2):
		for y in range(Ny // 2, image.shape[1] - Ny // 2):
			resp[x, y] = 0
			for i in range(Nx):
				for j in range(Ny):
					resp[x, y] += img[x - Nx // 2 + i, y - Ny // 2 + j] * neigh[i, j]
			resp[x, y] = int(max(min_value, min(resp[x, y], max_value)))

	return resp


sobel_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

roberts_x = np.array([[-1, 0], [0, 1]])
roberts_y = np.array([[0, -1], [1, 0]])

filtro_a_norte = np.array([[1, 1, 1], [1, -2, 1], [-1, -1, -1]])
filtro_b_sul = np.array([[-1, -1, -1], [1, -2, 1], [1, 1, 1]])
filtro_c_leste = np.array([[-1, 1, 1], [-1, -2, 1], [-1, 1, 1]])
filtro_d_sudeste = np.array([[-1, -1, 1], [-1, -2, 1], [1, 1, 1]])

passa_baixa_3x3 = operacao_vizinhanca(image, passa_baixa(3))
passa_baixa_7x7 = operacao_vizinhanca(image, passa_baixa(7))

print("oi")
