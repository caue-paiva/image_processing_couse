"""
Trabalho 3 — Filtros Convolucionais e Transformada de Fourier
Processamento de Imagens

Parte I:  Filtros convolucionais (shift, caixa, gaussiano, laplace, sobel,
          sharpen laplace, unsharp mask, emboss)
Parte II: DFT 2D manual com reconstrução progressiva (baixas -> altas frequências)

Bibliotecas: numpy, imageio, matplotlib (visualização), os
"""

import os
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt


# ============================================================
# Diretórios
# ============================================================

DIR_IMAGENS = "imagens"
DIR_RESULTADOS = "resultados"
os.makedirs(DIR_RESULTADOS, exist_ok=True)

# Se False, não abre janelas; apenas salva arquivos em resultados/
SHOW_PLOTS = False


# ============================================================
# Funções Utilitárias
# ============================================================

def luminosity(img):
    """Converte RGB para escala de cinza via luminosidade perceptual."""
    if img.ndim == 2:
        return img
    return (0.21 * img[..., 0] + 0.72 * img[..., 1] + 0.07 * img[..., 2]).astype(np.uint8)


def norm_minmax(img, C=255):
    """Normaliza imagem para o intervalo [0, C]."""
    img = img.astype(float)
    mn, mx = img.min(), img.max()
    if mx - mn == 0:
        return np.zeros_like(img)
    return (img - mn) / (mx - mn) * C


def show_bw(img, title="", save_path=None):
    """Exibe imagem em escala de cinza e opcionalmente salva."""
    plt.figure()
    if title:
        plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def show_side_by_side(img1, img2, t1="Original", t2="Resultado", save_path=None):
    """Exibe duas imagens lado a lado."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title(t1)
    axes[0].axis('off')
    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title(t2)
    axes[1].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def compute_spectrum(img):
    """
    Retorna espectro de magnitude (log, centralizado) e fase.

    Observação: este trabalho não permite usar FFT pronta do numpy.
    Portanto, calculamos a DFT 2D manualmente (matricial) e aplicamos shift manual.
    Para manter viabilidade, o espectro é calculado sobre uma versão reduzida
    (por amostragem) quando a imagem é grande.
    """
    img = img.astype(float)
    img_small = downsample_for_dft(img, max_size=256)
    F = fftshift2(dft2d(img_small))
    magnitude = np.log2(np.abs(F) + 1e-6)
    phase = np.angle(F)
    return magnitude, phase


def show_spectrum(img, title="Espectro", save_path=None):
    """Exibe o espectro de magnitude de uma imagem."""
    mag, _ = compute_spectrum(img)
    show_bw(mag, title=title, save_path=save_path)


# ============================================================
# Convolução 2D
# ============================================================

def convolve(img, kernel, padding_mode='reflect'):
    """
    Convolução 2D com suporte a diferentes modos de padding.

    padding_mode:
        'none'    — sem padding, bordas ficam zero
        'zero'    — zero-padding
        'reflect' — reflexão nas bordas
        'wrap'    — circular (toroidal)
    """
    img = img.astype(float)
    kernel = np.flip(kernel)
    k_h, k_w = kernel.shape
    a = (k_h - 1) // 2
    b = (k_w - 1) // 2

    if padding_mode == 'none':
        # Sem padding: bordas não são processadas
        new_img = np.zeros_like(img)
        h, w = img.shape
        for i in range(a, h - a):
            for j in range(b, w - b):
                region = img[i - a:i + a + 1, j - b:j + b + 1]
                new_img[i, j] = (kernel * region).sum()
        return new_img

    # Modos com padding via np.pad
    pad_modes = {
        'zero': ('constant', {'constant_values': 0}),
        'reflect': ('reflect', {}),
        'wrap': ('wrap', {}),
    }
    mode, kwargs = pad_modes[padding_mode]
    img_pad = np.pad(img, ((a, a), (b, b)), mode=mode, **kwargs)

    h, w = img.shape
    new_img = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            region = img_pad[i:i + k_h, j:j + k_w]
            new_img[i, j] = (kernel * region).sum()
    return new_img


# ============================================================
# Kernels — Parte I
# ============================================================

def shift_kernel(k):
    """Kernel de deslocamento k x k. Desloca a imagem para cima-esquerda."""
    kernel = np.zeros((k, k))
    kernel[k - 1, k - 1] = 1
    return kernel


def box_kernel(size):
    """Filtro de média (caixa) size x size."""
    return np.ones((size, size)) / float(size ** 2)


def gaussian_kernel(size, sigma):
    """Filtro gaussiano 2D com desvio padrão sigma."""
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=float)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def laplace_kernel():
    """Kernel laplaciano 3x3 (segunda derivada isotrópica)."""
    return np.array([[1, 1, 1],
                     [1, -8, 1],
                     [1, 1, 1]])


def sobel_x_kernel():
    """Sobel para gradiente horizontal (detecta bordas verticais)."""
    return np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])


def sobel_y_kernel():
    """Sobel para gradiente vertical (detecta bordas horizontais)."""
    return np.array([[-1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1]])


def emboss_kernel():
    """Filtro de relevo (emboss) — efeito tridimensional."""
    return np.array([[-2, -1, 0],
                     [-1, 1, 1],
                     [0, 1, 2]])


# ============================================================
# Processos com Filtros
# ============================================================

def sobel_magnitude(img, padding_mode='reflect'):
    """Magnitude do gradiente de Sobel: |Gx| + |Gy|."""
    gx = convolve(img, sobel_x_kernel(), padding_mode)
    gy = convolve(img, sobel_y_kernel(), padding_mode)
    return np.abs(gx) + np.abs(gy)


def sharpen_laplace(img, alpha=0.5, padding_mode='reflect'):
    """Aumento de nitidez com Laplaciano: f_sharp = f - alpha * laplaciano(f)."""
    lap = convolve(img, laplace_kernel(), padding_mode)
    result = img.astype(float) - alpha * lap
    return norm_minmax(np.abs(result)).astype(np.uint8)


def sharpen_unsharp(img, size=5, sigma=2.0, alpha=1.5, padding_mode='reflect'):
    """
    Aumento de nitidez com máscara de des-nitidez (unsharp mask).
    f_sharp = f + alpha * (f - blur(f))
    """
    blurred = convolve(img, gaussian_kernel(size, sigma), padding_mode)
    detail = img.astype(float) - blurred
    result = img.astype(float) + alpha * detail
    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
# Análise no Domínio da Frequência
# ============================================================

def filter_frequency_response(kernel, img_shape):
    """
    Calcula a resposta em frequência de um kernel espacial.
    Faz padding do kernel para o tamanho da imagem e aplica DFT manual (matricial).
    Para viabilidade, limita o cálculo a um tamanho máximo (amostragem) se necessário.
    """
    padded = np.zeros(img_shape, dtype=float)
    kh, kw = kernel.shape
    padded[:kh, :kw] = kernel
    padded_small = downsample_for_dft(padded, max_size=256)
    F = fftshift2(dft2d(padded_small))
    return np.log2(np.abs(F) + 1e-6)


def show_filter_freq_response(kernel, img_shape, title="Resposta em Frequência", save_path=None):
    """Exibe a resposta em frequência de um filtro."""
    resp = filter_frequency_response(kernel, img_shape)
    show_bw(resp, title=title, save_path=save_path)


# ============================================================
# Demonstrações — Parte I
# ============================================================

def demo_padding(img, kernel, filter_name, save_prefix):
    """Compara o efeito de diferentes modos de padding para um filtro."""
    modes = ['none', 'zero', 'reflect', 'wrap']
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"{filter_name} — Comparação de Padding", fontsize=14)

    for ax, mode in zip(axes, modes):
        result = convolve(img, kernel, padding_mode=mode)
        ax.imshow(norm_minmax(result), cmap='gray')
        ax.set_title(mode)
        ax.axis('off')

    plt.tight_layout()
    path = os.path.join(DIR_RESULTADOS, f"{save_prefix}_padding.png")
    plt.savefig(path, bbox_inches='tight', dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    print(f"  Salvo: {path}")


def demo_frequency(img, kernel, filter_name, save_prefix):
    """Mostra espectro antes/depois do filtro e resposta em frequência do kernel."""
    result = convolve(img, kernel, padding_mode='reflect')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{filter_name} — Domínio da Frequência", fontsize=14)

    # Espectro original
    mag_orig, _ = compute_spectrum(img)
    axes[0].imshow(mag_orig, cmap='gray')
    axes[0].set_title("Espectro Original")
    axes[0].axis('off')

    # Resposta em frequência do filtro
    freq_resp = filter_frequency_response(kernel, img.shape)
    axes[1].imshow(freq_resp, cmap='gray')
    axes[1].set_title("Resposta do Filtro")
    axes[1].axis('off')

    # Espectro após filtragem
    mag_result, _ = compute_spectrum(result)
    axes[2].imshow(mag_result, cmap='gray')
    axes[2].set_title("Espectro Filtrado")
    axes[2].axis('off')

    plt.tight_layout()
    path = os.path.join(DIR_RESULTADOS, f"{save_prefix}_frequencia.png")
    plt.savefig(path, bbox_inches='tight', dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    print(f"  Salvo: {path}")


def demo_filter(img, kernel, filter_name, save_prefix):
    """Demonstração completa de um filtro: resultado, padding e frequência."""
    print(f"\n{'='*60}")
    print(f"  {filter_name}")
    print(f"{'='*60}")

    # Resultado visual
    result = convolve(img, kernel, padding_mode='reflect')
    result_show = norm_minmax(result).astype(np.uint8)
    path = os.path.join(DIR_RESULTADOS, f"{save_prefix}_resultado.png")
    show_side_by_side(img, result_show, "Original", filter_name, save_path=path)
    print(f"  Salvo: {path}")

    # Comparação de padding
    demo_padding(img, kernel, filter_name, save_prefix)

    # Análise na frequência
    demo_frequency(img, kernel, filter_name, save_prefix)


def demo_process(img, process_fn, process_name, save_prefix, **kwargs):
    """Demonstração de um processo (sobel magnitude, sharpen, unsharp)."""
    print(f"\n{'='*60}")
    print(f"  {process_name}")
    print(f"{'='*60}")

    # Comparação de padding para processos
    modes = ['none', 'zero', 'reflect', 'wrap']
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"{process_name} — Comparação de Padding", fontsize=14)
    for ax, mode in zip(axes, modes):
        result = process_fn(img, padding_mode=mode, **kwargs)
        ax.imshow(norm_minmax(result), cmap='gray')
        ax.set_title(mode)
        ax.axis('off')
    plt.tight_layout()
    path = os.path.join(DIR_RESULTADOS, f"{save_prefix}_padding.png")
    plt.savefig(path, bbox_inches='tight', dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    print(f"  Salvo: {path}")

    # Resultado com reflect (padrão)
    result = process_fn(img, padding_mode='reflect', **kwargs)
    result_show = norm_minmax(result).astype(np.uint8)
    path = os.path.join(DIR_RESULTADOS, f"{save_prefix}_resultado.png")
    show_side_by_side(img, result_show, "Original", process_name, save_path=path)
    print(f"  Salvo: {path}")

    # Espectro antes/depois
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{process_name} — Espectro Antes e Depois", fontsize=14)
    mag_orig, _ = compute_spectrum(img)
    axes[0].imshow(mag_orig, cmap='gray')
    axes[0].set_title("Espectro Original")
    axes[0].axis('off')
    mag_result, _ = compute_spectrum(result)
    axes[1].imshow(mag_result, cmap='gray')
    axes[1].set_title("Espectro Filtrado")
    axes[1].axis('off')
    plt.tight_layout()
    path = os.path.join(DIR_RESULTADOS, f"{save_prefix}_frequencia.png")
    plt.savefig(path, bbox_inches='tight', dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    print(f"  Salvo: {path}")


def run_parte1(img_path):
    """Executa todas as demonstrações da Parte I."""
    print("\n" + "=" * 60)
    print("  PARTE I — FILTROS CONVOLUCIONAIS")
    print("=" * 60)

    img = luminosity(iio.imread(img_path))
    show_bw(img, "Imagem Original",
            save_path=os.path.join(DIR_RESULTADOS, "original.png"))

    # Filtros simples (kernel direto)
    filtros = [
        (shift_kernel(11), "Shift 11x11", "01_shift"),
        (box_kernel(7), "Caixa/Média 7x7", "02_box"),
        (gaussian_kernel(7, 2.0), "Gaussiano 7x7 sigma=2", "03_gaussiano"),
        (laplace_kernel(), "Laplaciano 3x3", "04_laplace"),
        (emboss_kernel(), "Emboss (Relevo) 3x3", "08_emboss"),
    ]

    for kernel, name, prefix in filtros:
        demo_filter(img, kernel, name, prefix)

    # Sobel (magnitude de dois kernels)
    demo_process(img, sobel_magnitude, "Sobel (Magnitude)", "05_sobel")

    # Frequência do Sobel — mostramos os dois kernels
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Sobel — Resposta em Frequência (Gx e Gy)", fontsize=14)
    axes[0].imshow(filter_frequency_response(sobel_x_kernel(), img.shape), cmap='gray')
    axes[0].set_title("Sobel X (horizontal)")
    axes[0].axis('off')
    axes[1].imshow(filter_frequency_response(sobel_y_kernel(), img.shape), cmap='gray')
    axes[1].set_title("Sobel Y (vertical)")
    axes[1].axis('off')
    plt.tight_layout()
    path = os.path.join(DIR_RESULTADOS, "05_sobel_freq_kernels.png")
    plt.savefig(path, bbox_inches='tight', dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

    # Sharpen com Laplace
    demo_process(img, sharpen_laplace, "Sharpen (Laplace, alpha=0.5)",
                 "06_sharpen_laplace", alpha=0.5)

    # Unsharp Mask
    demo_process(img, sharpen_unsharp, "Unsharp Mask (sigma=2, alpha=1.5)",
                 "07_unsharp_mask", size=5, sigma=2.0, alpha=1.5)


# ============================================================
# Parte II — Transformada de Fourier e Reconstrução Progressiva
# ============================================================

def downsample_for_dft(img, max_size=256):
    """
    Reduz a imagem por amostragem para tornar DFT manual viável.
    Mantém proporção e nunca aumenta a imagem.
    """
    h, w = img.shape
    scale = min(max_size / h, max_size / w, 1.0)
    if scale >= 1.0:
        return img
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    rows = np.linspace(0, h - 1, new_h).astype(int)
    cols = np.linspace(0, w - 1, new_w).astype(int)
    return img[np.ix_(rows, cols)]


def fftshift2(arr):
    """Versão 2D de fftshift (sem usar np.fft)."""
    h, w = arr.shape
    return np.roll(np.roll(arr, h // 2, axis=0), w // 2, axis=1)


def ifftshift2(arr):
    """Versão 2D de ifftshift (sem usar np.fft)."""
    h, w = arr.shape
    return np.roll(np.roll(arr, -(h // 2), axis=0), -(w // 2), axis=1)


def dft2d(img):
    """
    DFT 2D manual via fórmula direta.
    F(u,v) = sum_x sum_y f(x,y) * exp(-j*2*pi*(u*x/M + v*y/N))

    Usa imagem reduzida para viabilidade computacional.
    """
    img = img.astype(np.complex128, copy=False)
    M, N = img.shape

    # Forma matricial separável:
    # F = W_M @ img @ W_N
    x = np.arange(M)
    u = x[:, None]
    Wm = np.exp(-1j * 2 * np.pi * (u * x[None, :]) / M)

    y = np.arange(N)
    v = y[:, None]
    Wn = np.exp(-1j * 2 * np.pi * (v * y[None, :]) / N)

    return Wm @ img @ Wn.T


def idft2d(F):
    """
    IDFT 2D manual via fórmula direta.
    f(x,y) = (1/MN) sum_u sum_v F(u,v) * exp(j*2*pi*(u*x/M + v*y/N))
    """
    F = F.astype(np.complex128, copy=False)
    M, N = F.shape

    # Forma matricial separável:
    # f = (1/MN) * (W_M^{-1} @ F @ W_N^{-1})
    # onde W^{-1} = exp(+j*2*pi*x*u/N)
    u = np.arange(M)
    x = u[:, None]
    Wm_inv = np.exp(1j * 2 * np.pi * (x * u[None, :]) / M)

    v = np.arange(N)
    y = v[:, None]
    Wn_inv = np.exp(1j * 2 * np.pi * (y * v[None, :]) / N)

    img = (Wm_inv @ F @ Wn_inv.T) / (M * N)
    return img.real


def idft2d_progressive(F_shifted, n_steps=10):
    """
    Reconstrói imagem parcialmente, das baixas para altas frequências.
    Usa máscara circular crescente no espectro centralizado (fftshift).
    Retorna lista de (raio, imagem_parcial).
    """
    H, W = F_shifted.shape
    cy, cx = H // 2, W // 2

    # Matriz de distâncias ao centro
    u_coords = np.arange(H) - cy
    v_coords = np.arange(W) - cx
    uu, vv = np.meshgrid(v_coords, u_coords)
    distances = np.sqrt(uu ** 2 + vv ** 2)
    max_dist = distances.max()

    results = []
    for step in range(1, n_steps + 1):
        radius = (step / n_steps) * max_dist
        mask = (distances <= radius).astype(float)
        F_partial = F_shifted * mask
        # Reconstrução manual: desfaz shift e aplica IDFT 2D (matricial)
        F_unshifted = ifftshift2(F_partial)
        img_partial = idft2d(F_unshifted)
        results.append((radius, img_partial))

    return results


def run_parte2(img_high_path, img_low_path, n_steps=10, max_size=128):
    """
    Executa a Parte II: reconstrução progressiva da inversa de Fourier.
    Usa imagens redimensionadas para viabilidade da DFT manual.

    img_high_path: imagem com altas frequências dominantes (fogos, confetti)
    img_low_path:  imagem com baixas frequências dominantes (céu, paisagem)
    """
    print("\n" + "=" * 60)
    print("  PARTE II — TRANSFORMADA DE FOURIER E RECONSTRUÇÃO")
    print("=" * 60)

    for label, img_path in [("alta_freq", img_high_path), ("baixa_freq", img_low_path)]:
        print(f"\n--- Imagem: {label} ({img_path}) ---")

        img = luminosity(iio.imread(img_path))

        # Reduzir para viabilidade da DFT manual
        h, w = img.shape
        scale = min(max_size / h, max_size / w, 1.0)
        if scale < 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            # Redimensionamento simples por amostragem
            rows = np.linspace(0, h - 1, new_h).astype(int)
            cols = np.linspace(0, w - 1, new_w).astype(int)
            img_small = img[np.ix_(rows, cols)]
        else:
            img_small = img

        print(f"  Tamanho usado: {img_small.shape}")

        # DFT manual (matricial) — imagem reduzida para viabilidade
        print("  Calculando DFT 2D manual...")
        F = dft2d(img_small)
        F_shifted = fftshift2(F)

        # Espectro de magnitude
        mag = np.log2(np.abs(F_shifted) + 1e-6)
        path = os.path.join(DIR_RESULTADOS, f"p2_{label}_espectro.png")
        show_side_by_side(img_small, mag, "Original", "Espectro (DFT manual)", save_path=path)
        print(f"  Salvo: {path}")

        # Reconstrução progressiva
        print("  Reconstruindo progressivamente...")
        results = idft2d_progressive(F_shifted, n_steps=n_steps)

        # Grid com todos os passos
        cols_grid = min(5, n_steps)
        rows_grid = (n_steps + cols_grid - 1) // cols_grid
        fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(4 * cols_grid, 4 * rows_grid))
        fig.suptitle(f"Reconstrução Progressiva — {label}", fontsize=14)
        axes_flat = axes.flatten() if n_steps > 1 else [axes]

        for idx, (radius, img_partial) in enumerate(results):
            axes_flat[idx].imshow(img_partial, cmap='gray')
            axes_flat[idx].set_title(f"r={radius:.0f} ({(idx+1)*100//n_steps}%)")
            axes_flat[idx].axis('off')

        # Esconder eixos vazios
        for idx in range(len(results), len(axes_flat)):
            axes_flat[idx].axis('off')

        plt.tight_layout()
        path = os.path.join(DIR_RESULTADOS, f"p2_{label}_progressivo.png")
        plt.savefig(path, bbox_inches='tight', dpi=150)
        if SHOW_PLOTS:
            plt.show()
        plt.close(fig)
        print(f"  Salvo: {path}")

        # Verificação: reconstrução completa vs original
        print("  Calculando IDFT 2D manual (verificação)...")
        img_reconstructed = idft2d(F)
        diff = np.abs(img_small.astype(float) - img_reconstructed).max()
        print(f"  Erro máximo (original vs reconstruída): {diff:.6f}")

        path = os.path.join(DIR_RESULTADOS, f"p2_{label}_reconstruida.png")
        show_side_by_side(img_small, img_reconstructed,
                          "Original", "Reconstruída (IDFT manual)", save_path=path)
        print(f"  Salvo: {path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # --- Parte I ---
    # Coloque uma imagem em imagens/ para usar como entrada
    img_parte1 = os.path.join(DIR_IMAGENS, "foto.jpg")

    if os.path.exists(img_parte1):
        run_parte1(img_parte1)
    else:
        print(f"[Parte I] Imagem não encontrada: {img_parte1}")
        print("  Coloque uma imagem chamada 'foto.jpg' na pasta imagens/")

    # --- Parte II ---
    img_alta = os.path.join(DIR_IMAGENS, "alta_freq.jpg")
    img_baixa = os.path.join(DIR_IMAGENS, "baixa_freq.jpg")

    if os.path.exists(img_alta) and os.path.exists(img_baixa):
        # max_size=64 para DFT manual ser viável (64x64 = ~16M operações)
        run_parte2(img_alta, img_baixa, n_steps=10, max_size=64)
    else:
        print(f"\n[Parte II] Imagens não encontradas:")
        if not os.path.exists(img_alta):
            print(f"  - {img_alta} (altas frequências: fogos, confetti)")
        if not os.path.exists(img_baixa):
            print(f"  - {img_baixa} (baixas frequências: céu, paisagem)")
