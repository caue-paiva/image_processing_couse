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
# Sinusoides e a Exponencial Complexa
# ============================================================
# Desejamos funções com a propriedade f(t) = f(t + k*T)
# Senos e cosenos têm essa propriedade

x = np.arange(-5, 5, 0.2)

plt.figure()
plt.title("Seno e Cosseno (periodicidade)")
plt.scatter(x, np.sin(x + 2 * np.pi))
plt.scatter(x, np.cos(x + 2 * np.pi))
plt.show()

# O objeto periódico mais simples: círculo
circle = np.concatenate([np.sin(x)[:, None], np.cos(x)[:, None]], axis=1)
plt.figure(figsize=(4, 4))
plt.title("Círculo (sin, cos)")
plt.scatter(circle[:, 0], circle[:, 1])
plt.show()

# Soma de sinusoides com diferentes frequências
a = np.sin(x) + np.cos(x) + 0.2 * np.sin(2 * x) + 0.2 * np.cos(2 * x) + 0.05 * np.sin(10 * x) + 0.05 * np.cos(10 * x)
plt.figure()
plt.title("Soma de sinusoides")
plt.scatter(x, a)
plt.show()


# ============================================================
# Números complexos
# ============================================================

b = 1 + 3j
print("Parte imaginária de b:", b.imag)


# ============================================================
# Fórmula de Euler
# ============================================================
# Comparando resultado da exponencial com sen, cos

print("exp(0.1j) =", np.exp(0.1j))
print("sin(0.1) =", np.sin(0.1))
print("cos(0.1) =", np.cos(0.1))


# ============================================================
# Fundamentos para Transformada de Fourier
# ============================================================
# Diferentes sons/frequências e a soma delas

def make_sound(freq):
    x = np.arange(0, 1, 1 / 44000)
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

# Frequência única (Lá 440 Hz)
freq = 440
x_sound = np.arange(0, 1, 1 / 44000)
y_sound = np.sin(2 * np.pi * freq * x_sound)

plt.figure(figsize=(12, 3))
plt.title("Onda senoidal - 440 Hz")
plt.plot(x_sound, y_sound)
plt.show()

# Melodia: sequência de notas
y_melody = np.concatenate([
    make_sound(e4), make_sound(g4), make_sound(d4), make_sound(c4),
    make_sound(d4), make_sound(e4), make_sound(g4), make_sound(d4)
], axis=0)

# Acorde: soma de frequências (C major: C4 + E4 + G4)
y_chord = make_sound(c4) + make_sound(e4) + make_sound(g4)

plt.figure()
plt.title("Acorde C major (C4 + E4 + G4)")
plt.plot(x_sound, y_chord)
plt.show()

# Gostaríamos de criar um mecanismo para extração de frequências
# a partir de uma onda que soma diversas frequências
# -> Transformada de Fourier
