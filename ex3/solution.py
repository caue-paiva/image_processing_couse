import numpy as np
import imageio.v2 as imageio
import os

# Carrega a imagem
script_dir = os.path.dirname(os.path.abspath(__file__))
img = imageio.imread(os.path.join(script_dir, 'pompom.jpeg'))

# Converte para escala de cinza usando o metodo de luminosidade
# L = 0.299*R + 0.587*G + 0.114*B
img_gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]

# Aplica a FFT 2D
F = np.fft.fft2(img_gray)

# Obtem as magnitudes e ordena
magnitudes = np.abs(F)
sorted_mags = np.sort(magnitudes.flatten())

total_coeffs = F.size

# Busca binaria pelo limiar: encontra o maximo de coeficientes que podem
# ser zerados mantendo o erro quadratico medio <= 2.0
low, high = 0, total_coeffs - 1

while low < high:
    mid = (low + high + 1) // 2
    threshold = sorted_mags[mid]

    F_compressed = F.copy()
    F_compressed[magnitudes <= threshold] = 0

    img_r = np.real(np.fft.ifft2(F_compressed))
    mse = np.mean(np.sqrt((img_gray.astype(float) - img_r.astype(float))**2))

    if mse <= 2.0:
        low = mid
    else:
        high = mid - 1

# low eh o indice maximo onde o erro quadratico medio <= 2.0
threshold = sorted_mags[low]
F_compressed = F.copy()
F_compressed[magnitudes <= threshold] = 0

zeroed = np.sum(magnitudes <= threshold)
remaining = total_coeffs - zeroed

ratio = zeroed / total_coeffs
print(f"{ratio:.2f}")
