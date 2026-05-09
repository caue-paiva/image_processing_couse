# Processamento de Imagens — Filtros, Convoluções e Transformadas de Fourier

## Bibliotecas

- **numpy**: operações matriciais, FFT (`np.fft`), padding, manipulação de arrays
- **scipy**: filtros e operações complementares
- **imageio**: leitura/escrita de imagens (`imageio.v3 as iio`, `iio.imread`)
- **matplotlib**: visualização (não para processamento)

---

## 1. Fundamentos

### Conversão para escala de cinza (luminosidade)

```python
def luminosity(img):
    return (0.21 * img[..., 0] + 0.72 * img[..., 1] + 0.07 * img[..., 2]).astype(np.uint8)
```

### Normalização min-max

Mapeia valores para intervalo [0, C]:

```python
def norm_minmax(img, C=255):
    img = (img - img.min()) / (img.max() - img.min())
    return img * C
```

---

## 2. Convolução no Domínio Espacial

### Definição

C(x, y) = Σ_dx Σ_dy A(dx, dy) · B(x - dx, y - dy)

Onde A é o kernel (filtro) e B é a imagem. O kernel é espelhado (flip) antes da aplicação.

### Implementação

```python
def conv_op(i, j, img, kernel):
    k_h, k_w = kernel.shape
    a, b = (k_h - 1) // 2, (k_w - 1) // 2
    neighbourhood = img[i - a:i + a + 1, j - b:j + b + 1]
    return (kernel * neighbourhood).sum()

def convolve(img, kernel):
    img = img.astype(float)
    new_img = np.zeros_like(img)
    h, w = img.shape
    kernel = np.flip(kernel)
    k_h, k_w = kernel.shape
    a, b = (k_h - 1) // 2, (k_w - 1) // 2
    for i in range(a, h - a):
        for j in range(b, w - b):
            new_img[i, j] = conv_op(i, j, img, kernel)
    return new_img
```

Pixels nas bordas ficam sem processamento (borda = a ou b pixels).

---

## 3. Filtros no Domínio Espacial

### 3.1 Filtro de Média (Box)

Suavização simples. Cada pixel = média da vizinhança.

```python
def box_kernel(size):
    return np.ones((size, size)) / float(size**2)
```

### 3.2 Filtro Gaussiano

Suavização ponderada por distância ao centro. Preserva melhor bordas que o box.

G(x, y) = (1 / 2πσ²) · exp(-(x² + y²) / (2σ²))

```python
def gaussian_kernel(size, sigma):
    i_dir = np.arange(-1, 0.99, 2.0 / size)
    j_dir = np.arange(-1, 0.99, 2.0 / size)
    kernel = np.zeros((size, size))
    for i, i_s in enumerate(i_dir):
        for j, j_s in enumerate(j_dir):
            kernel[i, j] = np.exp(-(i_s**2 + j_s**2) / (2 * sigma**2))
    return kernel / kernel.sum()
```

### 3.3 Filtro de Mediana

Não-linear. Substitui pixel pela mediana da vizinhança. Excelente para ruído sal-e-pimenta.

```python
def median_filter(image, k):
    new_img = np.zeros_like(image)
    h, w = image.shape
    a = k // 2
    img_pad = np.pad(image, a, mode='reflect')
    for i in range(h):
        for j in range(w):
            neighbourhood = img_pad[i:i + 2*a + 1, j:j + 2*a + 1]
            new_img[i, j] = np.median(neighbourhood)
    return new_img
```

### 3.4 Filtro Laplaciano

Detecta bordas (segunda derivada). Realça mudanças bruscas de intensidade.

```python
def laplace():
    return np.array([[1, 1, 1],
                     [1, -8, 1],
                     [1, 1, 1]])
```

### 3.5 Aumento de Nitidez (Sharpen)

Subtrai bordas (laplaciano) da imagem original para realçar detalhes:

f_sharp = f - α · ∇²f

```python
def sharpen(img, alpha=0.1):
    nimg = convolve(img, laplace())
    return norm_minmax(np.absolute(img.astype(float) - alpha * nimg)).astype(np.uint8)
```

### 3.6 Filtro de Sobel

Detecta bordas com gradiente direcional (horizontal e vertical):

```python
def sobel_j_kernel():  # bordas verticais (gradiente horizontal)
    return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

def sobel_i_kernel():  # bordas horizontais (gradiente vertical)
    return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
```

Magnitude do gradiente: G = sqrt(Gx² + Gy²) ou aproximação |Gx| + |Gy|.

### 3.7 Filtro de Shift

Desloca a imagem (delta de Kronecker deslocado):

```python
def shift_kernel(k):
    kernel = np.zeros((k, k))
    kernel[k - 1, k - 1] = 1
    return kernel
```

---

## 4. Transformada de Fourier

### 4.1 Conceito

Qualquer sinal pode ser decomposto em soma de senos e cossenos de diferentes frequências.

**Fórmula de Euler**: e^(jθ) = cos(θ) + j·sin(θ)

### 4.2 DFT 1D (Transformada Discreta)

F(ω) = Σ_{t=0}^{N-1} f(t) · e^{-j·2π·ω·t/N}

Inversa: f(t) = (1/N) Σ_{ω=0}^{N-1} F(ω) · e^{j·2π·ω·t/N}

```python
def ft(signal, rate):
    n_samples = len(signal)
    freqs = np.arange(0, n_samples, 1)
    total_time = n_samples // rate
    t_samples = np.arange(0, total_time, total_time / n_samples)
    transform = np.zeros_like(signal, dtype=np.complex128)
    for freq in freqs[:rate // 2]:
        for idx_t, t in enumerate(t_samples):
            transform[freq] += signal[idx_t] * np.exp(-1j * 2 * np.pi * freq * t)
    return transform
```

### 4.3 FFT (Fast Fourier Transform)

Algoritmo Cooley-Tukey: O(N log N) em vez de O(N²). Requer N = potência de 2.

Divide em índices pares e ímpares recursivamente:

```python
def fft(y):
    N = len(y)
    if N == 1:
        return y
    freqs = np.arange(N // 2)
    F_even = fft(y[::2])
    F_odd = fft(y[1::2])
    W = np.exp(-1j * 2 * np.pi * freqs / N)
    return np.concatenate([F_even + F_odd * W, F_even - F_odd * W])
```

### 4.4 DFT 2D (para imagens)

F(u, v) = (1/MN) Σ_x Σ_y f(x,y) · e^{-j·2π·(ux/N + vy/M)}

Implementação: aplicar FFT 1D em cada linha, depois em cada coluna:

```python
def fft2d(img):
    F_h = np.zeros_like(img, dtype=np.complex128)
    F = np.zeros_like(img, dtype=np.complex128)
    H, W = img.shape
    for i in range(H):
        F_h[i, :] = fft(img[i, :])
    for j in range(W):
        F[:, j] = fft(F_h[:, j])
    return F
```

Com numpy: `np.fft.fft2(img)`.

### 4.5 Espectro e Fase

- **Magnitude (espectro)**: |F(u,v)| — mostra quais frequências estão presentes
- **Fase**: φ(u,v) = arctan(Im/Re) — posição espacial das frequências
- **Visualização**: usar `np.log2(magnitude + ε)` para comprimir a faixa dinâmica
- **fftshift**: `np.fft.fftshift()` centraliza as baixas frequências no centro da imagem

```python
def show_fft(img, show_phase=False):
    f_img = np.fft.fftshift(np.fft.fft2(img))
    plt.imshow(np.log2(np.abs(f_img) + 0.000001), cmap='gray')
```

---

## 5. Filtragem no Domínio da Frequência

### Teorema da Convolução

Convolução no domínio espacial = multiplicação no domínio da frequência:

g(x,y) = f(x,y) * h(x,y) ↔ G(u,v) = F(u,v) · H(u,v)

Workflow:
1. Calcular FFT da imagem: F = fft2(img)
2. Calcular FFT do filtro (ou construir H diretamente)
3. Multiplicar: G = F · H
4. Inverter: g = ifft2(G)

### 5.1 Filtro Passa-Baixa Ideal

Elimina frequências acima de um raio D₀:

H(u,v) = 1 se D(u,v) ≤ D₀, senão 0

```python
def ideal_low_pass(shape, D0):
    H = np.zeros(shape)
    cy, cx = shape[0] // 2, shape[1] // 2
    for u in range(shape[0]):
        for v in range(shape[1]):
            if np.sqrt((u - cy)**2 + (v - cx)**2) <= D0:
                H[u, v] = 1
    return H
```

Efeito: suavização. Problema: ringing (oscilações nas bordas) por causa do corte abrupto.

### 5.2 Filtro Passa-Alta Ideal

H(u,v) = 1 - H_passa_baixa(u,v)

Efeito: realça bordas, remove componentes suaves.

### 5.3 Filtro Passa-Banda e Rejeita-Banda

- **Passa-Banda**: permite apenas frequências entre D₁ e D₂
- **Rejeita-Banda**: bloqueia frequências entre D₁ e D₂

### 5.4 Filtro Gaussiano na Frequência

H(u,v) = exp(-D(u,v)² / (2·D₀²))

Transição suave, sem ringing. Equivale a convolver com Gaussiana no espaço.

### 5.5 Filtro Laplaciano na Frequência

H(u,v) = -4π²(u² + v²)

Realça altas frequências (bordas). Implementação via multiplicação no domínio da frequência.

### Aplicação de filtro no domínio da frequência

```python
def apply_freq_filter(img, H):
    F = np.fft.fftshift(np.fft.fft2(img))
    G = F * H
    return np.real(np.fft.ifft2(np.fft.ifftshift(G)))
```

### Padding do filtro espacial para domínio da frequência

Para aplicar um filtro espacial via FFT, fazer padding para o tamanho da imagem:

```python
def pad_filter(kernel, img_shape):
    padded = np.zeros(img_shape)
    kh, kw = kernel.shape
    padded[:kh, :kw] = kernel
    return padded
```

---

## 6. Relações entre Domínios

| Domínio Espacial | Domínio da Frequência |
|---|---|
| Convolução (lento, kernel pequeno) | Multiplicação (rápido, qualquer tamanho) |
| Filtro box → suavização | Passa-baixa → remove altas frequências |
| Filtro gaussiano → suavização suave | Gaussiana na frequência → sem ringing |
| Laplaciano → detecção de bordas | Passa-alta → realça altas frequências |
| Sobel → gradiente direcional | Componentes direcionais no espectro |

**Quando usar cada domínio:**
- Espacial: kernels pequenos (3x3, 5x5), filtros não-lineares (mediana)
- Frequência: kernels grandes, análise espectral, filtragem seletiva de frequências

---

## 7. Padrões de Código com numpy

```python
# Leitura e preparação
img = iio.imread('imagem.jpeg')
img_gray = luminosity(img)

# Convolução espacial
result = convolve(img_gray, gaussian_kernel(5, 2))

# FFT 2D com numpy
F = np.fft.fft2(img_gray)
F_shifted = np.fft.fftshift(F)
magnitude = np.abs(F_shifted)
phase = np.angle(F_shifted)

# Filtragem na frequência
G = F_shifted * H  # H é o filtro na frequência
result = np.real(np.fft.ifft2(np.fft.ifftshift(G)))

# Reconstrução (inversa)
reconstructed = np.fft.ifft2(F)
```

### Tamanho para FFT

Para FFT recursiva (Cooley-Tukey), N deve ser potência de 2. Usar `np.pad` para ajustar:

```python
img = np.pad(img, ((pad_h, 0), (0, pad_w)))
```

`np.fft.fft2` aceita qualquer tamanho (usa algoritmo misto internamente).
