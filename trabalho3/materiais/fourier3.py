import imageio.v3 as iio
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt


# ============================================================
# Representações de Funções
# ============================================================

def f(x, a, b, c):
    return c + b * x + a * (x**2)


# Amostragem da função
x = np.arange(-10, 10)
y = f(x, 2, 1, -4)
print("y =", y)

plt.figure()
plt.title("Amostragem da função quadrática")
plt.scatter(x, y)
plt.show()


# ============================================================
# Construindo o sistema linear
# ============================================================

A = np.zeros((20, 3))
A[:, 0] = x**2
A[:, 1] = x**1
A[:, 2] = x**0
print("Matriz A:")
print(A)

print("y transposto:", y.transpose())


# ============================================================
# Resolvendo o sistema linear
# ============================================================

c_1 = linalg.inv(A.transpose() @ A)
print("(A^T A)^-1:")
print(c_1)

C = c_1 @ (A.transpose() @ y)
print("Coeficientes encontrados:", C)


# ============================================================
# Fundamentos para Transformada de Fourier
# ============================================================
# Diferentes sons/frequências e a soma delas

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

# Frequência única (440 Hz)
freq = 440
x_sound = np.arange(0, 1, 1 / 5000)
y_sound = np.sin(2 * np.pi * freq * x_sound)

# Melodia: sequência de notas
y_melody = np.concatenate([
    make_sound(e4), make_sound(g4), make_sound(d4), make_sound(c4),
    make_sound(d4), make_sound(e4), make_sound(g4), make_sound(d4)
], axis=0)

plt.figure(figsize=(12, 3))
plt.title("Melodia (subamostrada)")
plt.plot(y_melody[::500])
plt.show()

# Acorde: soma de frequências (C major: C4 + E4 + G4)
cm = make_sound(c4) + make_sound(e4) + make_sound(g4)

plt.figure()
plt.title("Acorde C major (C4 + E4 + G4)")
plt.plot(cm[:3000:10])
plt.show()


# ============================================================
# Multiplicação do sinal pela função circular
# ============================================================
# Nosso plano: multiplicar o sinal pela função bidimensional
# que define um círculo

def fun_(t):
    return np.sin(t * (1 / 10) * np.pi * 2) + 1


def circulo(t, freq):
    return fun_(t) * np.cos(t * freq * np.pi * 2), fun_(t) * np.sin(t * freq * np.pi * 2)


def circulo_sem_f(t, freq):
    return np.cos(t * freq * np.pi * 2), np.sin(t * freq * np.pi * 2)


x_circ, y_circ = circulo(np.linspace(0, 50, 1000), freq=1 / 8)
plt.figure(figsize=(6, 6))
plt.title("Sinal enrolado no círculo")
plt.scatter(x_circ, y_circ)
plt.scatter(*circulo_sem_f(np.linspace(0, 50, 1000), freq=1))
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.show()


# ============================================================
# Controlando a frequência de volta no círculo
# ============================================================
# Observamos o efeito em diferentes funções

signal = make_sound(4)
samples = np.arange(0, 1, 1 / 44000)

plt.figure(figsize=(10, 25))
freqs = np.arange(0, 8, 0.2)
n_rows = len(freqs) // 4
centers_of_mass = []
for i, freq in enumerate(freqs):
    x, y = signal * np.sin(freq * samples * 2 * np.pi), signal * np.cos(freq * samples * 2 * np.pi)
    plt.subplot(n_rows, 4, i + 1)
    plt.axis([-1, 1, -1, 1])
    plt.plot(x, y)
    plt.plot(x.mean(), y.mean(), 'ro')
    plt.title(f"Freq = {freq:.2f}")
    centers_of_mass.append(x.sum())
plt.tight_layout()
plt.show()


# ============================================================
# Sinal que mistura 3 frequências
# ============================================================
# Algo interessante acontece quando as frequências do círculo
# se alinham com a frequência do sinal

signal = make_sound(1) + make_sound(4) + make_sound(7)
samples = np.arange(0, 1, 1 / 44000)

plt.figure(figsize=(10, 25))
freqs = np.arange(0, 8, 0.2)
n_rows = len(freqs) // 4
centers_of_mass = []
for i, freq in enumerate(freqs):
    x, y = signal * np.sin(freq * samples * 2 * np.pi), signal * np.cos(freq * samples * 2 * np.pi)
    plt.subplot(n_rows, 4, i + 1)
    plt.axis([-1, 1, -1, 1])
    plt.plot(x, y)
    plt.plot(x.mean(), y.mean(), 'ro')
    plt.title(f"Freq = {freq:.2f}")
    centers_of_mass.append(x.sum())
plt.tight_layout()
plt.show()


# ============================================================
# Centro de massa ao longo das frequências
# ============================================================
# Picos de centro de massa exatamente nas frequências do sinal

plt.figure()
plt.title("Centro de massa x frequência")
plt.plot(freqs, centers_of_mass)
plt.show()


# ============================================================
# Aumentando a resolução das frequências
# ============================================================
# Observando centro de massa nos eixos x (cosseno) e y (seno)

freqs = np.arange(0, 600, 0.1)
signal = cm
centers_of_mass = []
for i, freq in enumerate(freqs):
    x, y = signal * np.cos(2 * np.pi * freq * samples), signal * np.sin(2 * np.pi * freq * samples)
    centers_of_mass.append([x.mean(), y.mean()])

plt.figure()
plt.title("Centro de massa (cos e sin) - alta resolução")
plt.plot(freqs, centers_of_mass)
plt.show()


# ============================================================
# Reconstrução do sinal (inversa)
# ============================================================
# Desfazendo a transformação desenrolando o círculo

centers_of_mass = np.array(centers_of_mass)
signal_reconstructed = np.zeros_like(signal)
for i, freq in enumerate(freqs):
    signal_reconstructed += (centers_of_mass[i, 0] * np.cos(2 * np.pi * freq * samples)
                             + centers_of_mass[i, 1] * np.sin(2 * np.pi * freq * samples))

print("Sinal reconstruído (primeiras 10 amostras):", signal_reconstructed[:10])


# ============================================================
# Transformada de Fourier (definição formal)
# ============================================================
# F(w) = sum_{t=0}^{N} f(t) * e^{-j*2*pi*w*t}
# f(t) = sum_{w=0}^{N} F(w) * e^{j*2*pi*w*t}

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


# Aplicando a FT na melodia (com rate menor para viabilidade)
f_sound = make_sound
lullaby = np.concatenate([
    f_sound(e4, 2000), f_sound(g4, 2000), f_sound(d4, 2000), f_sound(c4, 2000),
    f_sound(d4, 2000), f_sound(e4, 2000), f_sound(g4, 2000), f_sound(d4, 2000)
])
f_lullaby = ft(lullaby, 2000)

plt.figure()
plt.title("Transformada de Fourier (implementação manual)")
plt.plot(abs(f_lullaby)[0:1000])
plt.show()


# ============================================================
# Fast Fourier Transform (FFT)
# ============================================================
# Versão rápida permite usar mais amostras do sinal

plt.figure()
plt.title("FFT da melodia")
plt.plot(abs(np.fft.fft(lullaby)))
plt.show()
