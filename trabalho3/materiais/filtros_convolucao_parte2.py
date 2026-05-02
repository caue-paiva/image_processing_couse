import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Funções de Suporte
# ============================================================

def luminosity(img):
    """Converte imagem colorida para preto e branco usando luminosidade."""
    return (0.21 * img[..., 0] + 0.72 * img[..., 1] + 0.07 * img[..., 2]).astype(np.uint8)


def norm_minmax(img, C=255, m=0):
    """Normalização min-max para deixar imagens em [0-255]."""
    img = (img - img.min()) / (img.max() - img.min())
    img *= C
    img -= m
    return img


def show_bw(img, title=""):
    """Exibe imagem em escala de cinza."""
    plt.figure()
    if title:
        plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.show()


# ============================================================
# Função de Convolução
# ============================================================

def conv_op(i, j, img, kernel):
    """Operação de convolução pontual centrada em (i, j)."""
    k_h, k_w = kernel.shape
    a = (k_h - 1) // 2
    b = (k_w - 1) // 2

    neighbourhood = img[i - a:i + a + 1, j - b:j + b + 1]

    c_mul = kernel * neighbourhood
    return c_mul.sum()


def convolve(img, kernel):
    """Convolução 2D de uma imagem com um kernel."""
    img = img.astype(float)
    new_img = np.zeros_like(img)
    h, w = img.shape

    kernel = np.flip(kernel)
    k_h, k_w = kernel.shape
    a = (k_h - 1) // 2
    b = (k_w - 1) // 2

    for i in range(a, h - a):
        for j in range(b, w - b):
            new_img[i, j] = conv_op(i, j, img, kernel)

    return new_img


# ============================================================
# Definição de Filtros
# ============================================================

# --- Filtro de Shift ---
def shift_kernel(k):
    kernel = np.zeros((k, k))
    kernel[k - 1, k - 1] = 1
    return kernel


# --- Filtro de Caixa/Média ---
def box_kernel(size):
    return np.ones((size, size)) / float(size**2)


# --- Filtro Gaussiano ---
# G_2D(x, y, sigma) = (1 / 2*pi*sigma^2) * exp(-(x^2 + y^2) / (2*sigma^2))
def gaussian_kernel(size, sigma):
    i_dir = np.arange(-1, 0.99, 2.0 / size)
    j_dir = np.arange(-1, 0.99, 2.0 / size)

    kernel = np.zeros((size, size))
    for i, i_sample in enumerate(i_dir):
        for j, j_sample in enumerate(j_dir):
            kernel[i, j] = np.exp(-(i_sample**2 + j_sample**2) / (2 * (sigma**2)))
    kernel = kernel / kernel.sum()
    return kernel


# --- Filtro de Mediana ---
def median_filter(image, k):
    new_img = np.zeros_like(image)
    h, w = image.shape

    a = b = k // 2

    img_pad = np.pad(image, a, mode='reflect')

    for i in range(0, h):
        for j in range(0, w):
            neighbourhood = img_pad[i:i + 2 * a + 1, j:j + 2 * b + 1]
            new_img[i, j] = np.median(neighbourhood)

    return new_img


# --- Filtro de Laplace e Aumento de Nitidez ---
def laplace():
    return np.array([[1, 1, 1],
                     [1, -8, 1],
                     [1, 1, 1]])


def sharpen(img, alpha=0.1):
    nimg = convolve(img, laplace())
    return norm_minmax(np.absolute((img.astype(float) - alpha * nimg))).astype(np.uint8)


# --- Filtro de Sobel ---
def sobel_j_kernel():
    return np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]])


def sobel_i_kernel():
    return np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]])


# ============================================================
# Aplicação dos Filtros
# ============================================================

if __name__ == "__main__":
    # Carregar imagem e converter para preto e branco
    img = iio.imread('XT206484.jpeg')
    img = luminosity(img)
    show_bw(img, "Imagem Original")

    # Filtro de Shift (49x49)
    show_bw(convolve(img, shift_kernel(49)), "Filtro de Shift 49x49")

    # Filtro de Caixa/Média (13x13)
    show_bw(convolve(img, box_kernel(13)), "Filtro de Caixa/Média 13x13")

    # Filtro Gaussiano (5x5, sigma=2)
    show_bw(convolve(img, gaussian_kernel(5, 2)), "Filtro Gaussiano 5x5 sigma=2")

    # Filtro de Mediana (9x9)
    show_bw(median_filter(img, 9), "Filtro de Mediana 9x9")

    # --- Laplace e Sharpen (usando imagem da lua) ---
    img_moon = iio.imread('moon.png')
    img_moon = luminosity(img_moon)

    # Filtro de Laplace
    show_bw(convolve(img_moon, laplace()), "Filtro de Laplace (moon)")

    # Sharpen com alpha=0.5
    show_bw(sharpen(img_moon, 0.5), "Sharpen alpha=0.5 (moon)")
