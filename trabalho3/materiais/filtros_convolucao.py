import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Funções de Suporte
# ============================================================

def luminosity(img):
    """Converte imagem colorida para preto e branco usando luminosidade."""
    return (0.21 * img[..., 0] + 0.72 * img[..., 1] + 0.07 * img[..., 2]).astype(np.uint8)


def norm_minmax(img, C, m):
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
# Aplicação dos Filtros
# ============================================================

if __name__ == "__main__":
    # Carregar imagem e converter para preto e branco
    img = iio.imread('XT206484.jpeg')
    img = luminosity(img)
    show_bw(img, "Imagem Original")

    # Filtro de Média (7x7)
    kernel_mean = np.ones((7, 7)) / 49
    new_img = convolve(img, kernel_mean)
    show_bw(new_img, "Filtro de Média 7x7")

    # Filtro de Bordas (Sobel)
    kernel_sobel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]])
    new_img = convolve(img, kernel_sobel)
    show_bw(new_img, "Filtro Sobel")

    # Combinação de Média + Sobel
    new_img = convolve(convolve(img, kernel_mean), kernel_sobel)
    show_bw(new_img, "Combinação Média + Sobel")
