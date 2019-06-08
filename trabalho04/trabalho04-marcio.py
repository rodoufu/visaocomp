from matplotlib import pyplot as plt
import math
import numpy as np
import cv2

img_name = 'storm_trooper'
img = cv2.imread(img_name + '.jpg', cv2.IMREAD_GRAYSCALE)

plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.show()

height, width = img.shape
print("height: {}, width: {}".format(height, width))

# Manually chosen
drawing_points_image = np.zeros((height, width), np.uint8)
np.copyto(drawing_points_image, img)

from_points = [(525, 110), (1410, 225), (1250, 1310), (30, 1075)]
cv2.circle(drawing_points_image, from_points[0], 5, 255, -1)
cv2.circle(drawing_points_image, from_points[1], 5, 255, -1)
cv2.circle(drawing_points_image, from_points[2], 5, 255, -1)
cv2.circle(drawing_points_image, from_points[3], 5, 255, -1)

# changing coordinate system of points (opencv -> numpy)
from_points = [(110, 525), (225, 1410), (1310, 1250), (1075, 30)]

fig = plt.figure(figsize=(10, 10))

img1 = fig.add_subplot(2, 2, 1)
img1.axis([420, 620, 210, 10])
img1.set_title("Point 1 (top left corner)")
img1.imshow(drawing_points_image, cmap='gray')

img2 = fig.add_subplot(2, 2, 2)
img2.axis([1250, 1450, 320, 120])
img2.set_title("Point 2 (top right corner)")
img2.imshow(drawing_points_image, cmap='gray')

img3 = fig.add_subplot(2, 2, 3)
img3.axis([0, 200, 1170, 970])
img3.set_title("Point 4 (bottom left corner)")
img3.imshow(drawing_points_image, cmap='gray')

img4 = fig.add_subplot(2, 2, 4)
img4.axis([1140, 1340, 1400, 1200])
img4.set_title("Point 3 (bottom right corner)")
img4.imshow(drawing_points_image, cmap='gray')

plt.show()

# points on the final image
to_points = [(0, 0), (0, width), (width, height), (height, 0)]

A = np.zeros((2 * 4, 8))

for i in range(4):
	A[2 * i] = [from_points[i][0], from_points[i][1], 1, 0, 0, 0, \
	            -from_points[i][0] * to_points[i][0], -from_points[i][1] * to_points[i][0]]
	A[2 * i + 1] = [0, 0, 0, from_points[i][0], from_points[i][1], 1, \
	                -from_points[i][0] * to_points[i][1], -from_points[i][1] * to_points[i][1]]
A

b = [x for t in to_points for x in t]

H = np.linalg.lstsq(A, b, rcond=None)[0]
H = np.append(H, 1).reshape(3, 3)
print("b: " + str(b) + "\n")
print("H: " + str(H))

# Inverse of H
H_inv = np.linalg.inv(H)

# creating the empty images to save results
nearest_neighbor_interpol_image = np.zeros((height, width), np.uint8)
bilinear_interpol_image = np.zeros((height, width), np.uint8)

print("inverse H: \n" + str(H_inv))


# Auxiliary functions
def dist(p1, p2):
	return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def interpolate(v1, fv1, v2, fv2, vm):
	return ((v2 - vm) / (v2 - v1)) * fv1 + ((vm - v1) / (v2 - v1)) * fv2


def calc_bilinear(result, points, values):
	R1 = interpolate(points[0][0], values[0], points[1][0], values[1], result[0])
	R2 = interpolate(points[2][0], values[2], points[3][0], values[3], result[0])
	return interpolate(points[0][1], R1, points[3][1], R2, result[1])


# Calculating the final image
for px in range(0, height - 1):
	for py in range(0, width - 1):
		result = np.dot(H_inv, np.array([px, py, 1]))
		result = result / result[2]

		# Getting the four pixels around the one calculated
		x_floor = math.floor(result[0])
		x_ceil = math.ceil(result[0])
		y_floor = math.floor(result[1])
		y_ceil = math.floor(result[1])
		points = [(x_floor, y_floor), (x_floor, y_ceil), (x_ceil, y_ceil), (x_ceil, y_floor)]

		# Calculating the distance between the point and the surround pixels
		distances = [dist(result, points[0]), dist(result, points[1]), dist(result, points[2]), dist(result, points[3])]

		# Getting the pixel closest to the one calculated
		index_of_min = np.argmin(distances)
		nearest_neighbor_interpol_image[px, py] = img[points[index_of_min][0], points[index_of_min][1]]

		# results coordinates is exactly one point in the original image
		if (points[0][0] == points[1][0]) and points[2][0] == points[3][0]:
			bilinear_interpol_image[px, py] = img[int(result[0]), int(result[1])]

		else:
			# Getting the pixel closest to the one calculated
			values = [img[points[0] + (0,)], img[points[1] + (0,)], img[points[1] + (0,)], img[points[1] + (0,)]]
			bilinear_interpol_image[px, py] = calc_bilinear(result, points, values)
fig = plt.figure(figsize=(15, 15))

img1 = fig.add_subplot(1, 2, 1)
img1.set_title("Nearest Neighbor Interpolation")
img1.imshow(nearest_neighbor_interpol_image, cmap='gray')

img2 = fig.add_subplot(1, 2, 2)
img2.set_title("Bilinear Interpolation")
img2.imshow(bilinear_interpol_image, cmap='gray')

plt.show()

fig = plt.figure(figsize=(20, 20))

img1 = fig.add_subplot(1, 2, 1)
img1.axis([300, 500, 750, 600])
img1.set_title("Nearest Neighbor Interpolation")
img1.imshow(nearest_neighbor_interpol_image, cmap='gray')

img2 = fig.add_subplot(1, 2, 2)
img2.axis([300, 500, 750, 600])
img2.set_title("Bilinear Interpolation")
img2.imshow(bilinear_interpol_image, cmap='gray')

plt.show()
