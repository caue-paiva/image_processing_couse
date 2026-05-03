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
from scipy.signal import convolve2d
import math


# ============================================================
# Diretórios
# ============================================================

DIR_IMAGENS = "imagens"
DIR_RESULTADOS = "resultados"
os.makedirs(DIR_RESULTADOS, exist_ok=True)

# Se False, não abre janelas; apenas salva arquivos em resultados/
SHOW_PLOTS = False

# Limite para DFT manual usada em espectros/resposta em frequência (Parte I).
# Mantém os gráficos rápidos mesmo para imagens grandes.
DFT_MAX_SIZE = 96

# Tamanho do recorte (pixels) para visualizar diferenças de padding nas bordas
PADDING_CROP = 500


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


def compute_spectrum(img, max_size=DFT_MAX_SIZE):
    """
    Retorna espectro de magnitude (log, centralizado) e fase.

    Observação: este trabalho não permite usar FFT pronta do numpy.
    Portanto, calculamos a DFT 2D manualmente (matricial) e aplicamos shift manual.
    Para manter viabilidade, o espectro é calculado sobre uma versão reduzida
    (por amostragem) quando a imagem é grande.
    """
    img = img.astype(float)
    img_small = downsample_for_dft(img, max_size=max_size)
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
    k_h, k_w = kernel.shape
    a = (k_h - 1) // 2
    b = (k_w - 1) // 2

    def _box_filter_numpy(image, kh, kw, pad_mode):
        """
        Box filter (kernel constante) via imagem integral (O(H*W)).
        Suporta padding via np.pad: constant/reflect/wrap.

        Para tamanhos pares, usamos padding assimétrico para imitar o alinhamento
        de 'same' (top/left bias): before=a, after=kh-1-a.
        """
        pad_top = a
        pad_left = b
        pad_bottom = (kh - 1 - a)
        pad_right = (kw - 1 - b)

        if pad_mode == 'zero':
            padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)),
                            mode='constant', constant_values=0)
        elif pad_mode == 'reflect':
            padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)),
                            mode='reflect')
        elif pad_mode == 'wrap':
            padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)),
                            mode='wrap')
        else:
            raise ValueError(pad_mode)

        # imagem integral com borda 0 para facilitar diferenças
        S = np.pad(padded, ((1, 0), (1, 0)), mode='constant', constant_values=0).cumsum(0).cumsum(1)

        H, W = image.shape
        y0 = np.arange(0, H)
        x0 = np.arange(0, W)
        y1 = y0 + kh
        x1 = x0 + kw

        # somas por broadcast: (H,1) e (1,W)
        A = S[y1[:, None], x1[None, :]]
        B = S[y0[:, None], x1[None, :]]
        C = S[y1[:, None], x0[None, :]]
        D = S[y0[:, None], x0[None, :]]
        return A - B - C + D

    # Fast-path: kernel tipo "shift" (delta) — evita O(H*W*k^2) do convolve2d
    # Detecta kernel com um único valor não-zero.
    nz = np.argwhere(kernel != 0)
    if nz.shape[0] == 1:
        (ku, kv) = nz[0]
        delta = float(kernel[ku, kv])
        if abs(delta) > 0:
            # Para um delta em (ku,kv), a convolução (com flip interno do convolve2d)
            # equivale a deslocar a imagem em (k_h-1-ku, k_w-1-kv) pixels para cima-esquerda.
            # Ex.: shift_kernel(k) coloca delta em (0,0) => desloca (k-1,k-1).
            du = (k_h - 1 - ku)
            dv = (k_w - 1 - kv)

            h, w = img.shape
            out = np.zeros((h, w), dtype=float)

            if padding_mode in ('none', 'zero'):
                # Região que consegue "puxar" pixels válidos sem sair dos limites
                src_r0, src_r1 = du, h
                src_c0, src_c1 = dv, w
                dst_r0, dst_r1 = 0, h - du
                dst_c0, dst_c1 = 0, w - dv
                if dst_r1 > dst_r0 and dst_c1 > dst_c0:
                    out[dst_r0:dst_r1, dst_c0:dst_c1] = img[src_r0:src_r1, src_c0:src_c1] * delta
                if padding_mode == 'none':
                    # Manter comportamento "borda não processada" equivalente ao kernel:
                    # zera uma faixa de a/b pixels (apesar de shift não ser centrado, isso
                    # é coerente com o resto do trabalho).
                    if a > 0:
                        out[:a, :] = 0
                        out[-a:, :] = 0
                    if b > 0:
                        out[:, :b] = 0
                        out[:, -b:] = 0
                return out

            # reflect / wrap: usa padding e faz slicing (rápido)
            pad_map = {'reflect': 'reflect', 'wrap': 'wrap'}
            if padding_mode in pad_map:
                img_pad = np.pad(img, ((du, 0), (dv, 0)), mode=pad_map[padding_mode])
                return img_pad[:h, :w] * delta

            raise ValueError(f"padding_mode inválido: {padding_mode!r}")

    # Fast-path: Box filter (kernel constante) via numpy (imagem integral)
    if np.all(kernel == kernel.flat[0]):
        if padding_mode == 'none':
            out = _box_filter_numpy(img, k_h, k_w, pad_mode='zero') * float(kernel.flat[0])
            # manter comportamento de borda "não processada"
            if a > 0:
                out[:a, :] = 0
                out[-a:, :] = 0
            if b > 0:
                out[:, :b] = 0
                out[:, -b:] = 0
            return out
        if padding_mode in ('zero', 'reflect', 'wrap'):
            return _box_filter_numpy(img, k_h, k_w, pad_mode=padding_mode) * float(kernel.flat[0])
        raise ValueError(f"padding_mode inválido: {padding_mode!r}")

    if padding_mode == 'none':
        # Calcula com zero-padding e depois zera as bordas para manter
        # o comportamento original (borda "não processada").
        new_img = convolve2d(img, kernel, mode='same', boundary='fill', fillvalue=0)
        if a > 0:
            new_img[:a, :] = 0
            new_img[-a:, :] = 0
        if b > 0:
            new_img[:, :b] = 0
            new_img[:, -b:] = 0
        return new_img

    if padding_mode == 'zero':
        return convolve2d(img, kernel, mode='same', boundary='fill', fillvalue=0)
    if padding_mode == 'reflect':
        # SciPy usa 'symm' para reflexão/simetria nas bordas
        return convolve2d(img, kernel, mode='same', boundary='symm')
    if padding_mode == 'wrap':
        return convolve2d(img, kernel, mode='same', boundary='wrap')

    raise ValueError(f"padding_mode inválido: {padding_mode!r}")


# ============================================================
# Kernels — Parte I
# ============================================================

def shift_kernel(k, shift=None):
    """
    Kernel de deslocamento k x k (delta).

    Por padrão, desloca em (k-1) pixels para cima-esquerda (delta no canto superior-esquerdo).
    Se `shift` for informado, desloca exatamente `shift` pixels (delta em [shift, shift]).
    """
    kernel = np.zeros((k, k))
    if shift is None:
        kernel[0, 0] = 1
    else:
        if shift < 0 or shift >= k:
            raise ValueError(f"shift inválido: {shift} (precisa 0 <= shift < k)")
        kernel[shift, shift] = 1
    return kernel


def box_kernel(size):
    """Filtro de média (caixa) size x size."""
    return np.ones((size, size)) / float(size ** 2)


def gaussian_kernel(size, sigma):
    """Filtro gaussiano 2D com desvio padrão sigma."""
    # Suporta tamanhos pares e ímpares mantendo exatamente (size x size).
    # Coordenadas centradas em (size-1)/2.
    ax = np.arange(size, dtype=float) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax, indexing='xy')
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def laplace_kernel(size=3):
    """
    Kernel laplaciano.

    - size=3: Laplaciano 8-vizinhos (clássico isotrópico)
    - size=9/18/26/36: Laplaciano "maior" (multi-escala) construído por suavização binomial
      e 2ª derivada discreta (separável).
    """
    if size == 3:
        return np.array([[1, 1, 1],
                         [1, -8, 1],
                         [1, 1, 1]], dtype=float)
    if size not in (9, 18, 26, 36):
        raise ValueError(f"size inválido para laplace_kernel: {size} (use 3, 9, 18, 26 ou 36)")

    # Suavização binomial (linha size-1 do triângulo de Pascal) e 2ª derivada discreta
    n = size - 1
    s = np.array([math.comb(n, k) for k in range(size)], dtype=float)
    d2_full = np.convolve(s, np.array([1.0, -2.0, 1.0]), mode='full')
    start = (len(d2_full) - size) // 2
    d2 = d2_full[start:start + size]

    K = np.outer(s, d2) + np.outer(d2, s)
    m = np.max(np.abs(K))
    if m > 0:
        K = K / m
    return K


def sobel_x_kernel(size=13):
    """
    Sobel para gradiente horizontal (detecta bordas verticais).

    Por padrão usamos a versão 13x13 para que as diferenças de padding fiquem
    mais evidentes no relatório. Passe size=3 para o Sobel clássico 3x3.
    """
    if size == 3:
        return np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    if size == 7:
        # Construção separável 7x7: suavização (binomial) x derivada discreta
        s = np.array([1, 6, 15, 20, 15, 6, 1], dtype=float)
        d = np.array([-1, -4, -5, 0, 5, 4, 1], dtype=float)
        return np.outer(s, d)
    if size == 13:
        # Construção separável 13x13 (hardcoded):
        # s = coeficientes binomiais (linha 12 do triângulo de Pascal)
        # d = vetor derivativo anti-simétrico (realça variações horizontais)
        s = np.array([1, 12, 66, 220, 495, 792, 924, 792, 495, 220, 66, 12, 1], dtype=float)
        d = np.array([-1, -12, -66, -220, -495, -792, 0, 792, 495, 220, 66, 12, 1], dtype=float)
        return np.outer(s, d)
    raise ValueError(f"size inválido para sobel_x_kernel: {size} (use 3, 7 ou 13)")


def sobel_y_kernel(size=13):
    """
    Sobel para gradiente vertical (detecta bordas horizontais).

    Por padrão usamos a versão 13x13; passe size=3 para o Sobel clássico 3x3.
    """
    if size == 3:
        return np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])
    if size == 7:
        s = np.array([1, 6, 15, 20, 15, 6, 1], dtype=float)
        d = np.array([-1, -4, -5, 0, 5, 4, 1], dtype=float)
        return np.outer(d, s)
    if size == 13:
        s = np.array([1, 12, 66, 220, 495, 792, 924, 792, 495, 220, 66, 12, 1], dtype=float)
        d = np.array([-1, -12, -66, -220, -495, -792, 0, 792, 495, 220, 66, 12, 1], dtype=float)
        return np.outer(d, s)
    raise ValueError(f"size inválido para sobel_y_kernel: {size} (use 3, 7 ou 13)")


def emboss_kernel():
    """Filtro de relevo (emboss) — efeito tridimensional."""
    return np.array([[-2, -1, 0],
                     [-1, 1, 1],
                     [0, 1, 2]])


# ============================================================
# Processos com Filtros
# ============================================================

def sobel_magnitude(img, padding_mode='reflect', size=13):
    """Magnitude do gradiente de Sobel: |Gx| + |Gy|."""
    gx = convolve(img, sobel_x_kernel(size=size), padding_mode)
    gy = convolve(img, sobel_y_kernel(size=size), padding_mode)
    return np.abs(gx) + np.abs(gy)


def sharpen_laplace(img, alpha=0.5, padding_mode='reflect', laplace_size=36):
    """Aumento de nitidez com Laplaciano: f_sharp = f - alpha * laplaciano(f)."""
    lap = convolve(img, laplace_kernel(size=laplace_size), padding_mode)
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
    Faz padding do kernel e aplica DFT manual (matricial).

    Importante: não fazemos downsample do kernel após colocá-lo em um canvas do
    tamanho da imagem, pois a amostragem pode "pular" o suporte do kernel e
    zerá-lo (ex.: shift 11x11 com 1 em [10,10]). Em vez disso, calculamos a
    resposta em frequência em um canvas pequeno (DFT_MAX_SIZE) e colocamos o
    kernel centralizado nele.
    """
    kh, kw = kernel.shape
    H, W = img_shape
    Hc = min(DFT_MAX_SIZE, H)
    Wc = min(DFT_MAX_SIZE, W)

    padded_small = np.zeros((Hc, Wc), dtype=float)

    # Coloca o kernel centralizado no canvas pequeno.
    # Se o kernel for maior que o canvas, recorta (situação improvável aqui).
    kh_use = min(kh, Hc)
    kw_use = min(kw, Wc)
    k_crop = kernel[:kh_use, :kw_use]

    top = (Hc - kh_use) // 2
    left = (Wc - kw_use) // 2
    padded_small[top:top + kh_use, left:left + kw_use] = k_crop

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
    results = []
    for mode in modes:
        results.append(convolve(img, kernel, padding_mode=mode))

    # Usar a mesma escala em todos os subplots para a diferença ser visível
    mn = float(min(r.min() for r in results))
    mx = float(max(r.max() for r in results))
    if mx - mn < 1e-12:
        vmin, vmax = mn, mn + 1.0
    else:
        vmin, vmax = mn, mx

    # --- Figura 1: imagem inteira (mantém o comportamento atual) ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"{filter_name} — Comparação de Padding", fontsize=14)
    for ax, mode, result in zip(axes, modes, results):
        ax.imshow(result, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(mode)
        ax.axis('off')

    plt.tight_layout()
    path = os.path.join(DIR_RESULTADOS, f"{save_prefix}_padding.png")
    plt.savefig(path, bbox_inches='tight', dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    print(f"  Salvo: {path}")

    # --- Figura 2: recortes dos cantos (diferenças de padding ficam evidentes) ---
    h, w = img.shape
    crop = int(min(PADDING_CROP, h, w))
    if crop < 8:
        return

    corners = [
        ("TL", (slice(0, crop), slice(0, crop))),
        ("TR", (slice(0, crop), slice(w - crop, w))),
        ("BL", (slice(h - crop, h), slice(0, crop))),
        ("BR", (slice(h - crop, h), slice(w - crop, w))),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    fig.suptitle(f"{filter_name} — Padding (Cantos, crop={crop}px)", fontsize=14)

    for r, (corner_label, (rs, cs)) in enumerate(corners):
        row_crops = [res[rs, cs] for res in results]
        row_min = float(min(c.min() for c in row_crops))
        row_max = float(max(c.max() for c in row_crops))
        if row_max - row_min < 1e-12:
            row_vmin, row_vmax = row_min, row_min + 1.0
        else:
            row_vmin, row_vmax = row_min, row_max

        for c, (mode, crop_img) in enumerate(zip(modes, row_crops)):
            ax = axes[r, c]
            ax.imshow(crop_img, cmap='gray', vmin=row_vmin, vmax=row_vmax)
            if r == 0:
                ax.set_title(mode)
            if c == 0:
                ax.set_ylabel(corner_label, rotation=0, labelpad=20, va='center')
            ax.axis('off')

    plt.tight_layout()
    path = os.path.join(DIR_RESULTADOS, f"{save_prefix}_padding_cantos.png")
    plt.savefig(path, bbox_inches='tight', dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    print(f"  Salvo: {path}")


def demo_frequency(img, kernel, filter_name, save_prefix, mag_orig=None):
    """Mostra espectro antes/depois do filtro e resposta em frequência do kernel."""
    result = convolve(img, kernel, padding_mode='reflect')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{filter_name} — Domínio da Frequência", fontsize=14)

    # Espectro original
    if mag_orig is None:
        mag_orig, _ = compute_spectrum(img)

    # Espectro após filtragem
    mag_result, _ = compute_spectrum(result)

    # Usar a MESMA escala para os espectros antes/depois, para a diferença ficar visível
    vmin = min(float(mag_orig.min()), float(mag_result.min()))
    vmax = max(float(mag_orig.max()), float(mag_result.max()))

    axes[0].imshow(mag_orig, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title("Espectro Original")
    axes[0].axis('off')

    # Resposta em frequência do filtro
    freq_resp = filter_frequency_response(kernel, img.shape)
    axes[1].imshow(freq_resp, cmap='gray')
    axes[1].set_title("Resposta do Filtro")
    axes[1].axis('off')

    axes[2].imshow(mag_result, cmap='gray', vmin=vmin, vmax=vmax)
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


def demo_process(img, process_fn, process_name, save_prefix, mag_orig=None, **kwargs):
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
    if mag_orig is None:
        mag_orig, _ = compute_spectrum(img)
    mag_result, _ = compute_spectrum(result)
    vmin = min(float(mag_orig.min()), float(mag_result.min()))
    vmax = max(float(mag_orig.max()), float(mag_result.max()))
    axes[0].imshow(mag_orig, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title("Espectro Original")
    axes[0].axis('off')
    axes[1].imshow(mag_result, cmap='gray', vmin=vmin, vmax=vmax)
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

    # Cache: espectro original (DFT manual é caro; reaproveitar em todas as demos)
    mag_orig, _ = compute_spectrum(img)

    # Filtros simples (kernel direto)
    filtros = [
        (shift_kernel(111), "Shift 111x111", "01_shift"),
        (box_kernel(33), "Caixa/Média 33x33", "02_box"),
        (gaussian_kernel(28, 3.0), "Gaussiano 28x28 sigma=4", "03_gaussiano"),
        (laplace_kernel(3), "Laplaciano 3x3", "04_laplace"),
        (emboss_kernel(), "Emboss (Relevo) 3x3", "08_emboss"),
    ]

    for kernel, name, prefix in filtros:
        # demo_filter chama demo_frequency; passamos o espectro original para evitar recomputar
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        result = convolve(img, kernel, padding_mode='reflect')
        result_show = norm_minmax(result).astype(np.uint8)
        path = os.path.join(DIR_RESULTADOS, f"{prefix}_resultado.png")
        show_side_by_side(img, result_show, "Original", name, save_path=path)
        print(f"  Salvo: {path}")

        demo_padding(img, kernel, name, prefix)
        demo_frequency(img, kernel, name, prefix, mag_orig=mag_orig)

    # Sobel (magnitude) — usando kernel 13x13
    demo_process(img, sobel_magnitude, "Sobel 13x13 (Magnitude)", "05_sobel", mag_orig=mag_orig, size=13)

    # Frequência do Sobel — mostramos os dois kernels
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Sobel 13x13 — Resposta em Frequência (Gx e Gy)", fontsize=14)
    axes[0].imshow(filter_frequency_response(sobel_x_kernel(size=13), img.shape), cmap='gray')
    axes[0].set_title("Sobel X (horizontal)")
    axes[0].axis('off')
    axes[1].imshow(filter_frequency_response(sobel_y_kernel(size=13), img.shape), cmap='gray')
    axes[1].set_title("Sobel Y (vertical)")
    axes[1].axis('off')
    plt.tight_layout()
    path = os.path.join(DIR_RESULTADOS, "05_sobel_freq_kernels.png")
    plt.savefig(path, bbox_inches='tight', dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

    # Sharpen com Laplace
    demo_process(img, sharpen_laplace, "Sharpen (Laplace 36x36, alpha=0.5)",
                 "06_sharpen_laplace", mag_orig=mag_orig, alpha=0.5, laplace_size=36)

    # Unsharp Mask (blur maior para evidenciar efeitos de padding)
    demo_process(img, sharpen_unsharp, "Unsharp Mask (size=24, sigma=6, alpha=1.5)",
                 "07_unsharp_mask", mag_orig=mag_orig, size=24, sigma=6.0, alpha=1.5)


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
