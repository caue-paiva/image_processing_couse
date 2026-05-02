import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Funções de Suporte
# ============================================================

def luminosity(img):
    return (0.21 * img[..., 0] + 0.72 * img[..., 1] + 0.07 * img[..., 2]).astype(np.uint8)


def norm_minmax(img, C=255, m=0):
    img = (img - img.min()) / (img.max() - img.min())
    img *= C
    img -= m
    return img


def show_bw(img, title=""):
    plt.figure()
    if title:
        plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.show()


def show_bw2(ogimg, img):
    plt.figure(figsize=(10, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(ogimg, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(img, cmap='gray')
    plt.show()


def show_fft(ogimg, show_phase=False):
    plt.figure(figsize=(15, 30))
    f_img = np.log2(np.fft.fftshift(np.fft.fft2(ogimg)) + 0.000001).real
    plt.subplot(1, 2 if not show_phase else 3, 1)
    plt.imshow(ogimg, cmap='gray')
    plt.subplot(1, 2 if not show_phase else 3, 2)
    plt.imshow(f_img, cmap='gray')
    if show_phase:
        f_img = np.fft.fft2(ogimg)
        plt.subplot(1, 3, 3)
        plt.imshow(np.arctan2(f_img.imag, f_img.real), cmap='gray')
    plt.show()


# ============================================================
# Transformada Rápida de Fourier 1D
# ============================================================

def make_sound(freq, rate=44000):
    x = np.arange(0, 1, 1 / rate)
    y = np.sin(2 * np.pi * freq * x)
    return y


# Notas musicais
c4 = 261.63
d4 = 293.66
e4 = 329.63
f4 = 349.23
g4 = 392.00
a4 = 440.00
b4 = 493.88

rate = 2**14
y = np.concatenate([
    make_sound(e4, rate=rate), make_sound(g4, rate=rate), make_sound(d4, rate=rate),
    make_sound(c4, rate=rate), make_sound(d4, rate=rate), make_sound(e4, rate=rate),
    make_sound(g4, rate=rate), make_sound(d4, rate=rate)
], axis=0)


def fft(y):
    """Implementação recursiva da FFT (Cooley-Tukey)."""
    N = len(y)

    if N == 1:
        return y

    freqs = np.arange(N // 2)

    F_even = fft(y[::2])
    F_odd = fft(y[1::2])
    W = np.exp(-1j * 2 * np.pi * freqs / N)

    return np.concatenate([F_even + F_odd * W, F_even - F_odd * W])


plt.figure()
plt.title("FFT 1D da melodia (implementação manual)")
plt.plot(np.abs(fft(y))[0:5000])
plt.show()


# ============================================================
# Transformada Rápida de Fourier 2D
# ============================================================

def fft2d(img):
    """FFT 2D aplicando FFT 1D em cada linha e depois em cada coluna."""
    F_h = np.zeros_like(img, dtype=np.complex128)
    F = np.zeros_like(img, dtype=np.complex128)
    H, W = img.shape
    for i in range(H):
        F_h[i, :] = fft(img[i, :])
    for j in range(W):
        F[:, j] = fft(F_h[:, j])
    return F


# Aplicação em imagem
img = iio.imread('pompom_brownie.jpeg')[0:256, 0:256, :]
img = luminosity(img)
img = np.pad(img, ((256 - img.shape[0], 0), (0, 0)))

plt.figure()
plt.title("FFT 2D (implementação manual)")
plt.imshow(np.log2(np.fft.fftshift(np.abs(fft2d(img)) + 0.0001)), cmap='gray')
plt.show()

# Comparação com np.fft.fft2
show_fft(img)
