"""
Editor de Imagens - Trabalho 01
SCC0251 - Processamento de Imagens
Aluno: Cauê Paiva Lira - 14675416
Data: Semestre 1 de 2026

Título:
Image CLIpper - Ferramenta CLI para manipular imagens
"""

import argparse
import numpy as np
import imageio.v2 as imageio


# ---------------------------------------------------------------------------
# Helpers de I/O
# ---------------------------------------------------------------------------

def load_image(path):
    """Carrega imagem e converte para float64."""
    img = imageio.imread(path)
    return img.astype(np.float64)


def save_image(img, path):
    """Clip [0,255], cast uint8 e salva."""
    img = np.clip(img, 0, 255).astype(np.uint8)
    imageio.imwrite(path, img)


# ---------------------------------------------------------------------------
# Interpolacao bilinear (usado por rotate e scale)
# ---------------------------------------------------------------------------

def _bilinear_interp(img, y_coords, x_coords, mode='constant', cval=0.0):
    """Interpolacao bilinear para coordenadas fracionarias.

    img: imagem 2D (H,W) ou 3D (H,W,C)
    y_coords, x_coords: arrays de coordenadas float (mesma shape)
    mode: 'constant' (preenche cval), 'nearest' (estende borda),
          'wrap' (wrap-around)
    Retorna array com mesma shape que y_coords (+ canais se RGB).
    """
    h, w = img.shape[:2]
    is_rgb = img.ndim == 3

    y0 = np.floor(y_coords).astype(np.int64)
    x0 = np.floor(x_coords).astype(np.int64)
    y1 = y0 + 1
    x1 = x0 + 1

    fy = y_coords - y0.astype(np.float64)
    fx = x_coords - x0.astype(np.float64)

    if mode == 'wrap':
        y0 = y0 % h
        y1 = y1 % h
        x0 = x0 % w
        x1 = x1 % w
    elif mode == 'nearest':
        y0 = np.clip(y0, 0, h - 1)
        y1 = np.clip(y1, 0, h - 1)
        x0 = np.clip(x0, 0, w - 1)
        x1 = np.clip(x1, 0, w - 1)
    else:  # constant
        # Fora dos limites se a coordenada original esta fora da imagem
        out_of_bounds = ((y_coords < 0) | (y_coords > h - 1) |
                         (x_coords < 0) | (x_coords > w - 1))
        y0 = np.clip(y0, 0, h - 1)
        y1 = np.clip(y1, 0, h - 1)
        x0 = np.clip(x0, 0, w - 1)
        x1 = np.clip(x1, 0, w - 1)

    # Valores nos 4 vizinhos
    v00 = img[y0, x0]
    v01 = img[y0, x1]
    v10 = img[y1, x0]
    v11 = img[y1, x1]

    if is_rgb:
        fy = fy[..., np.newaxis]
        fx = fx[..., np.newaxis]

    result = (v00 * (1 - fy) * (1 - fx) +
              v01 * (1 - fy) * fx +
              v10 * fy * (1 - fx) +
              v11 * fy * fx)

    if mode == 'constant':
        if is_rgb:
            result[out_of_bounds] = cval
        else:
            result[out_of_bounds] = cval

    return result


# ---------------------------------------------------------------------------
# Transformacoes Geometricas
# ---------------------------------------------------------------------------

def translate(img, dx, dy):
    """Translacao com wrap-around usando numpy.roll."""
    result = np.roll(img, int(dy), axis=0)
    result = np.roll(result, int(dx), axis=1)
    return result


def detect_black_pixels(original, rotated):
    """Detecta pixels pretos novos introduzidos pela rotacao."""
    if original.ndim == 3:
        was_black = np.all(original == 0, axis=2)
        is_black = np.all(rotated == 0, axis=2)
    else:
        was_black = (original == 0)
        is_black = (rotated == 0)
    new_black = is_black & ~was_black
    count = int(np.count_nonzero(new_black))
    pct = 100.0 * count / new_black.size
    return count > 0, count, pct


def compute_autozoom_factor(angle_deg, h, w):
    """Calcula fator de zoom para eliminar bordas pretas na rotacao."""
    angle_rad = np.deg2rad(abs(angle_deg))
    abs_a = angle_rad % (np.pi / 2)
    cos_a = np.cos(abs_a)
    sin_a = np.sin(abs_a)
    zoom_h = (h * cos_a + w * sin_a) / h
    zoom_w = (h * sin_a + w * cos_a) / w
    return max(zoom_h, zoom_w)


def _rotate_core(img, angle, mode='constant', cval=0.0):
    """Rotacao em torno do centro com interpolacao bilinear.

    angle: graus (sentido anti-horario)
    mode: 'constant' ou 'nearest'
    Retorna imagem com mesmas dimensoes (reshape=False).
    """
    h, w = img.shape[:2]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    rad = np.deg2rad(angle)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)

    # Grid de coordenadas de saida
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

    # Mapeamento inverso: para cada pixel de saida, achar origem
    src_x = cos_a * (xx - cx) + sin_a * (yy - cy) + cx
    src_y = -sin_a * (xx - cx) + cos_a * (yy - cy) + cy

    return _bilinear_interp(img, src_y, src_x, mode=mode, cval=cval)


def rotate_image(img, angle, strategy):
    """Rotacao em torno do centro sem pixels pretos.

    strategy: 'autozoom' ou 'nearest'
    """
    if strategy == 'nearest':
        return _rotate_core(img, angle, mode='nearest')

    # autozoom
    h, w = img.shape[:2]
    zf = compute_autozoom_factor(angle, h, w)
    zoomed = scale_image(img, zf)
    rotated = _rotate_core(zoomed, angle, mode='constant', cval=0.0)
    # Recortar ao tamanho original (centro)
    rh, rw = rotated.shape[:2]
    start_y = (rh - h) // 2
    start_x = (rw - w) // 2
    return rotated[start_y:start_y + h, start_x:start_x + w]


def scale_image(img, factor):
    """Escala por fator com interpolacao bilinear."""
    h, w = img.shape[:2]
    new_h = int(round(h * factor))
    new_w = int(round(w * factor))

    # Grid de coordenadas de saida mapeadas para a imagem original
    yy = np.linspace(0, h - 1, new_h)
    xx = np.linspace(0, w - 1, new_w)
    grid_y, grid_x = np.meshgrid(yy, xx, indexing='ij')

    return _bilinear_interp(img, grid_y, grid_x, mode='constant', cval=0.0)


def crop_image(img, x, y, w, h):
    """Recorte via slicing numpy."""
    return img[y:y + h, x:x + w]


# ---------------------------------------------------------------------------
# Transformacoes de Intensidade
# ---------------------------------------------------------------------------

def inverse(img):
    """Inversao de intensidade."""
    return 255.0 - img


def log_transform(img):
    """Transformacao logaritmica."""
    c = 255.0 / np.log(256.0)
    return c * np.log(1.0 + img)


def gamma_transform(img, gamma):
    """Correcao gamma."""
    return 255.0 * (img / 255.0) ** gamma


def contrast_transform(img, intervals, targets):
    """Mapeamento piecewise linear de contraste."""
    return np.interp(img, intervals, targets)


def solarize(img, threshold):
    """Solarizacao: inverte tons acima do limiar."""
    return np.where(img < threshold, img, 255.0 - img)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description='Editor de Imagens - SCC0251 Trabalho 01')
    sub = parser.add_subparsers(dest='command')
    sub.required = True

    # Flags comuns
    def add_io(p):
        p.add_argument('-i', '--input', required=True, help='Imagem de entrada')
        p.add_argument('-o', '--output', required=True, help='Imagem de saida')

    # translate
    p = sub.add_parser('translate', help='Translacao')
    add_io(p)
    p.add_argument('--dx', type=int, required=True)
    p.add_argument('--dy', type=int, required=True)

    # rotate
    p = sub.add_parser('rotate', help='Rotacao')
    add_io(p)
    p.add_argument('--angle', type=float, required=True)
    p.add_argument('--strategy', choices=['autozoom', 'nearest'],
                   required=True, help='Estrategia para pixels pretos')

    # scale
    p = sub.add_parser('scale', help='Escala')
    add_io(p)
    p.add_argument('--factor', type=float, required=True)

    # crop
    p = sub.add_parser('crop', help='Recorte')
    add_io(p)
    p.add_argument('--x', type=int, required=True)
    p.add_argument('--y', type=int, required=True)
    p.add_argument('--w', type=int, required=True)
    p.add_argument('--h', type=int, required=True)

    # inverse
    p = sub.add_parser('inverse', help='Inversao de intensidade')
    add_io(p)

    # log
    p = sub.add_parser('log', help='Transformacao logaritmica')
    add_io(p)

    # gamma
    p = sub.add_parser('gamma', help='Correcao gamma')
    add_io(p)
    p.add_argument('--gamma', type=float, required=True)

    # contrast
    p = sub.add_parser('contrast', help='Mapeamento piecewise de contraste')
    add_io(p)
    p.add_argument('--intervals', type=str, required=True,
                   help='Valores de entrada separados por virgula')
    p.add_argument('--targets', type=str, required=True,
                   help='Valores de saida separados por virgula')

    # creative
    p = sub.add_parser('creative', help='Solarizacao (transformacao criativa)')
    add_io(p)
    p.add_argument('--threshold', type=float, required=True)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    img = load_image(args.input)

    if args.command == 'translate':
        result = translate(img, args.dx, args.dy)

    elif args.command == 'rotate':
        result = rotate_image(img, args.angle, strategy=args.strategy)

    elif args.command == 'scale':
        result = scale_image(img, args.factor)

    elif args.command == 'crop':
        result = crop_image(img, args.x, args.y, args.w, args.h)

    elif args.command == 'inverse':
        result = inverse(img)

    elif args.command == 'log':
        result = log_transform(img)

    elif args.command == 'gamma':
        result = gamma_transform(img, args.gamma)

    elif args.command == 'contrast':
        intervals = [float(v) for v in args.intervals.split(',')]
        targets = [float(v) for v in args.targets.split(',')]
        result = contrast_transform(img, intervals, targets)

    elif args.command == 'creative':
        result = solarize(img, args.threshold)

    save_image(result, args.output)
    print(f"Imagem salva em {args.output}")


if __name__ == '__main__':
    main()
