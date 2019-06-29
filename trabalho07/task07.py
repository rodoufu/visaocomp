import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from sklearn.metrics import mean_squared_error


def image_side_by_side(imgs, titles):
    imgs = [np.round(img).astype("uint8") for img in imgs]
    plt.subplot(121), plt.imshow(imgs[0], cmap='gray'), plt.title(titles[0])
    plt.subplot(122), plt.imshow(imgs[1], cmap='gray'), plt.title(titles[1])
    plt.show()


def get_channel(image, channel_number):
    channel_cn = np.zeros(image.shape[:2])
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            channel_cn[x, y] = image[x, y][channel_number]
    return channel_cn


file_names = [f"LANDSAT_7_ETMXS_20000111_217_076_L2_BAND{i}.tif" for i in range(1, 4)]
file_name_reference = "LANDSAT_7_ETMPAN_20000111_217_076_L2_BAND8.tif"

# Opening the images
images = [cv2.imread(file_name, 0) for file_name in file_names]
image_reference = cv2.imread(file_name_reference, 0)

# Cropping
crop_size_1d = 512
crop_pos = [random.randrange(images[0].shape[0] - crop_size_1d), random.randrange(images[0].shape[1] - crop_size_1d)]

scale = 2
images = [img[crop_pos[0]:crop_pos[0] + crop_size_1d, crop_pos[1]:crop_pos[1] + crop_size_1d] for img in images]
image_reference = image_reference[
                  scale * crop_pos[0]:scale * (crop_pos[0] + crop_size_1d),
                  scale * crop_pos[1]:scale * (crop_pos[1] + crop_size_1d)]

# Creating composite
composite_image = np.zeros([images[0].shape[0], images[0].shape[1], 3], dtype="uint8")
for i in range(3):
    composite_image[..., i] = images[i]

# Resizing
new_size = images[0].shape[1] * scale, images[0].shape[0] * scale
# images = [cv2.resize(img, (int(new_size[0]), int(new_size[1]))) for img in images]
composite_image = cv2.resize(composite_image, (int(new_size[0]), int(new_size[1])))

# Converting to HSV
composite_image = cv2.cvtColor(composite_image, cv2.COLOR_BGR2HSV)

# Fusioning
composite_image[..., 2] = image_reference

# HSV to BGR
composite_image = cv2.cvtColor(composite_image, cv2.COLOR_HSV2BGR)
image_side_by_side([image_reference, composite_image], ["Referência", "Fusionado"])

composite_image = cv2.cvtColor(composite_image, cv2.COLOR_BGR2GRAY)
image_side_by_side([image_reference, composite_image], ["Referência", "Fusionado tons de cinza"])

print(f"Erro quadrático médio: {mean_squared_error(image_reference, composite_image):.3f}")
