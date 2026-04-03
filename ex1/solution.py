import imageio.v2 as imageio

img = imageio.imread('pompom_segredo.bmp')
h, w, _ = img.shape

# Canal R: row-major (linha por linha)
r_chars = ''.join(chr(v) for v in img[:, :, 0].flatten())
start = r_chars.find('segredo: ')
end = r_chars.find(';', start)
print(r_chars[start + 9:end])

# Canais G e B: column-major (coluna por coluna)
for ch in [1, 2]:
    chars = ''.join(chr(v) for v in img[:, :, ch].flatten(order='F'))
    start = chars.find('segredo: ')
    end = chars.find(';', start)
    print(chars[start + 9:end])
