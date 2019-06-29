import cv2
import numpy as np
from matplotlib import pyplot as plt


def fourier_magniture(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Imagem de entrada'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude do espectro de Fourier'), plt.xticks([]), plt.yticks([])
    plt.show()


estrela_topo = cv2.imread('estrelaTopo.png', 0)
fourier_magniture(estrela_topo)
estrela_fundo = cv2.imread('estrelaFundo.png', 0)
fourier_magniture(estrela_fundo)


def butterworth_filter(dft4img, stopband2=10, order=3, showdft=False):
    """
    Get Butterworth filter in frequency domain.
    """
    h, w = dft4img.shape[0], dft4img.shape[1]
    P = h / 2
    Q = w / 2
    dst = np.zeros((h, w, 3), np.float64)
    for i in range(h):
        for j in range(w):
            r2 = float((i - P) ** 2 + (j - Q) ** 2)
            if r2 == 0:
                r2 = 1.0
            dst[i, j] = 1 / (1 + (r2 / stopband2) ** order)
    dst = np.float64(dst)
    if showdft:
        cv2.imshow("butterworth", cv2.magnitude(dst[:, :, 0], dst[:, :, 1]))
    return dst


fourier_magniture(butterworth_filter(cv2.imread('estrelaTopo.png'), showdft=False))
