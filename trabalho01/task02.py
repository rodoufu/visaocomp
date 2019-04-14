import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def get_channel(image, channel_number):
    channel_cn = np.zeros(image.shape[:2])
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            channel_cn[x, y] = image[x, y][channel_number]
    return channel_cn


file_name = "VisaoTrab1.tiff"
file_name_no_extension = os.path.splitext(file_name)[0]

img = cv2.imread(file_name)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

gamma = 1024.0
channel_max = 179.0

channel_h = get_channel(hsv, 0) / channel_max
channel_h = (channel_h ** gamma) * channel_max


def set_channel(dst, src, channel_number, channel_min = 0, channel_max = 255):
    for x in range(dst.shape[0]):
        for y in range(dst.shape[1]):
            dst[x, y][channel_number] = int(max(min(src[x, y], channel_max), 0))


# set_channel(hsv, channel_h, 0, 0, channel_max)
for x in range(hsv.shape[0]):
    for y in range(hsv.shape[1]):
        hsv[x, y][0] = int(max(min(channel_h[x, y], channel_max), 0))

img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
file_name_gamma = f'{file_name_no_extension}_{gamma:.2f}.tiff'
cv2.imwrite(file_name_gamma, img)

mpl.rc("savefig", dpi = 150)

image_titles = [(file_name, 'Original'), (file_name_gamma, 'Imagem gamma')]
images_rows_cols = []

for image_file_name, title in image_titles:
    img = cv2.imread(image_file_name, 0)
    rows, cols = img.shape
    images_rows_cols += [(img, rows, cols)]
    plt.imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
    plt.title(title)
    plt.show()

for i in range(len(image_titles)):
    img, rows, cols = images_rows_cols[i]
    image_file_name, title = image_titles[i]

    hist, bins = np.histogram(img, range=(0, 255), bins=64)
    hist = hist / float(rows * cols)
    center = (bins[:-1] + bins[1:]) / 2
    first_not_zero = 0
    for i in range(len(hist)):
        first_not_zero = i
        if hist[i] > 0:
            break
    plt.bar(center, hist, align='center', color='m', width=4)
    plt.xlim([0, 255])
    plt.ylim([0, 0.1])
    plt.ylabel('$p(f(x,y))$', fontsize=16)
    plt.xlabel(f'Intensidade {title}', fontsize=16)
    m_y = 0.005
    m = np.mean(img)
    s = np.std(img)
    plt.plot(m, m_y , "ko")
    plt.plot([m - s, m + s], [m_y] * 2, "k--");
    plt.savefig(f'{os.path.splitext(image_file_name)[0]}_Hist.eps')
    plt.show()
    print(f'\nMedia: {np.mean(img):.20f}')
    print(f'Desvio Padrao: {np.std(img):.20f}')
    print(f'Primeiro valor não nulo no histograma: {first_not_zero}')
