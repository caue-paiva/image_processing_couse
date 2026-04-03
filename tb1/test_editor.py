"""
Testes de integracao para editor.py
Invoca o CLI via subprocess e compara imagens resultantes.
"""

import subprocess
import sys
import numpy as np
import numpy.testing as npt
import imageio.v2 as imageio
import pytest

EDITOR = "tb1/editor.py"
PYTHON = sys.executable


def run_editor(*args):
    """Executa editor.py e retorna CompletedProcess."""
    return subprocess.run(
        [PYTHON, EDITOR, *args],
        capture_output=True, text=True
    )


def make_gradient_gray(shape=(10, 10)):
    """Cria imagem grayscale com gradiente."""
    h, w = shape
    img = np.arange(h * w, dtype=np.uint8).reshape(h, w)
    # Garante que nao comeca em 0 para evitar falsos positivos em testes
    img = np.clip(img + 10, 0, 255).astype(np.uint8)
    return img


def make_gradient_rgb(shape=(10, 10)):
    """Cria imagem RGB com gradiente."""
    gray = make_gradient_gray(shape)
    return np.stack([gray, gray // 2, 255 - gray], axis=2).astype(np.uint8)


@pytest.fixture
def gray_img(tmp_path):
    """Salva imagem grayscale de teste e retorna (path, array)."""
    img = make_gradient_gray()
    path = str(tmp_path / "gray.png")
    imageio.imwrite(path, img)
    return path, img.astype(np.float64)


@pytest.fixture
def rgb_img(tmp_path):
    """Salva imagem RGB de teste e retorna (path, array)."""
    img = make_gradient_rgb()
    path = str(tmp_path / "rgb.png")
    imageio.imwrite(path, img)
    return path, img.astype(np.float64)


# ---------------------------------------------------------------------------
# Translacao
# ---------------------------------------------------------------------------

class TestTranslate:
    def test_identity(self, gray_img, tmp_path):
        inp, original = gray_img
        out = str(tmp_path / "out.png")
        r = run_editor('translate', '-i', inp, '-o', out, '--dx', '0', '--dy', '0')
        assert r.returncode == 0
        result = imageio.imread(out).astype(np.float64)
        npt.assert_array_almost_equal(result, original, decimal=0)

    def test_roundtrip(self, gray_img, tmp_path):
        inp, original = gray_img
        mid = str(tmp_path / "mid.png")
        out = str(tmp_path / "out.png")
        run_editor('translate', '-i', inp, '-o', mid, '--dx', '3', '--dy', '2')
        run_editor('translate', '-i', mid, '-o', out, '--dx', '-3', '--dy', '-2')
        result = imageio.imread(out).astype(np.float64)
        npt.assert_array_almost_equal(result, original, decimal=0)

    def test_no_black_pixels(self, gray_img, tmp_path):
        inp, original = gray_img
        out = str(tmp_path / "out.png")
        run_editor('translate', '-i', inp, '-o', out, '--dx', '5', '--dy', '5')
        result = imageio.imread(out)
        # Com wrap, nao deve haver pixels pretos que nao existiam
        was_black = (original == 0)
        is_black = (result.astype(np.float64) == 0)
        new_black = is_black & ~was_black
        assert not np.any(new_black)


# ---------------------------------------------------------------------------
# Rotacao
# ---------------------------------------------------------------------------

class TestRotate:
    def test_0deg(self, gray_img, tmp_path):
        inp, original = gray_img
        out = str(tmp_path / "out.png")
        r = run_editor('rotate', '-i', inp, '-o', out,
                        '--angle', '0', '--force')
        assert r.returncode == 0
        result = imageio.imread(out).astype(np.float64)
        npt.assert_array_almost_equal(result, original, decimal=0)

    def test_90deg(self, gray_img, tmp_path):
        inp, original = gray_img
        out = str(tmp_path / "out.png")
        r = run_editor('rotate', '-i', inp, '-o', out,
                        '--angle', '90', '--force')
        assert r.returncode == 0
        result = imageio.imread(out).astype(np.float64)
        # Rotacao 90 graus CCW = transpor + flip vertical
        expected = np.flipud(original.T) if original.ndim == 2 else original
        # Comparar dimensoes (reshape=False mantém tamanho original)
        assert result.shape == original.shape

    def test_autozoom_no_black(self, gray_img, tmp_path):
        inp, original = gray_img
        out = str(tmp_path / "out.png")
        r = run_editor('rotate', '-i', inp, '-o', out,
                        '--angle', '30', '--strategy', 'autozoom')
        assert r.returncode == 0
        result = imageio.imread(out).astype(np.float64)
        # Autozoom deve eliminar ou minimizar pixels pretos novos
        if original.ndim == 3:
            is_black = np.all(result == 0, axis=2)
        else:
            is_black = (result == 0)
        # Permite uma tolerancia pequena (bordas de arredondamento)
        black_pct = 100.0 * np.count_nonzero(is_black) / is_black.size
        assert black_pct < 5.0, f"Muitos pixels pretos: {black_pct:.1f}%"

    def test_nearest(self, gray_img, tmp_path):
        inp, _ = gray_img
        out = str(tmp_path / "out.png")
        r = run_editor('rotate', '-i', inp, '-o', out,
                        '--angle', '45', '--strategy', 'nearest')
        assert r.returncode == 0
        result = imageio.imread(out)
        assert result.shape[0] > 0 and result.shape[1] > 0


# ---------------------------------------------------------------------------
# Escala
# ---------------------------------------------------------------------------

class TestScale:
    def test_2x(self, gray_img, tmp_path):
        inp, original = gray_img
        out = str(tmp_path / "out.png")
        r = run_editor('scale', '-i', inp, '-o', out, '--factor', '2.0')
        assert r.returncode == 0
        result = imageio.imread(out)
        assert result.shape[0] == original.shape[0] * 2
        assert result.shape[1] == original.shape[1] * 2

    def test_half(self, gray_img, tmp_path):
        inp, original = gray_img
        out = str(tmp_path / "out.png")
        r = run_editor('scale', '-i', inp, '-o', out, '--factor', '0.5')
        assert r.returncode == 0
        result = imageio.imread(out)
        assert result.shape[0] == original.shape[0] // 2
        assert result.shape[1] == original.shape[1] // 2


# ---------------------------------------------------------------------------
# Crop
# ---------------------------------------------------------------------------

class TestCrop:
    def test_valid(self, gray_img, tmp_path):
        inp, original = gray_img
        out = str(tmp_path / "out.png")
        r = run_editor('crop', '-i', inp, '-o', out,
                        '--x', '2', '--y', '3', '--w', '5', '--h', '4')
        assert r.returncode == 0
        result = imageio.imread(out).astype(np.float64)
        expected = original[3:7, 2:7]
        npt.assert_array_almost_equal(result, expected, decimal=0)


# ---------------------------------------------------------------------------
# Transformacoes de Intensidade
# ---------------------------------------------------------------------------

class TestInverse:
    def test_twice_is_identity(self, gray_img, tmp_path):
        inp, original = gray_img
        mid = str(tmp_path / "mid.png")
        out = str(tmp_path / "out.png")
        run_editor('inverse', '-i', inp, '-o', mid)
        run_editor('inverse', '-i', mid, '-o', out)
        result = imageio.imread(out).astype(np.float64)
        npt.assert_array_almost_equal(result, original, decimal=0)


class TestLog:
    def test_range(self, gray_img, tmp_path):
        inp, _ = gray_img
        out = str(tmp_path / "out.png")
        r = run_editor('log', '-i', inp, '-o', out)
        assert r.returncode == 0
        result = imageio.imread(out).astype(np.float64)
        assert result.min() >= 0
        assert result.max() <= 255


class TestGamma:
    def test_identity(self, gray_img, tmp_path):
        inp, original = gray_img
        out = str(tmp_path / "out.png")
        r = run_editor('gamma', '-i', inp, '-o', out, '--gamma', '1.0')
        assert r.returncode == 0
        result = imageio.imread(out).astype(np.float64)
        npt.assert_array_almost_equal(result, original, decimal=0)


class TestContrast:
    def test_identity_mapping(self, gray_img, tmp_path):
        inp, original = gray_img
        out = str(tmp_path / "out.png")
        r = run_editor('contrast', '-i', inp, '-o', out,
                        '--intervals', '0,128,255',
                        '--targets', '0,128,255')
        assert r.returncode == 0
        result = imageio.imread(out).astype(np.float64)
        npt.assert_array_almost_equal(result, original, decimal=0)


class TestCreative:
    def test_solarize(self, gray_img, tmp_path):
        inp, original = gray_img
        out = str(tmp_path / "out.png")
        threshold = 50.0
        r = run_editor('creative', '-i', inp, '-o', out,
                        '--threshold', str(threshold))
        assert r.returncode == 0
        result = imageio.imread(out).astype(np.float64)
        # Pixels abaixo do limiar nao mudam
        below = original < threshold
        npt.assert_array_almost_equal(result[below], original[below], decimal=0)
        # Pixels acima do limiar sao invertidos
        above = original >= threshold
        expected_above = np.clip(255.0 - original[above], 0, 255)
        npt.assert_array_almost_equal(result[above], expected_above, decimal=0)


# ---------------------------------------------------------------------------
# Suporte RGB
# ---------------------------------------------------------------------------

class TestRGBSupport:
    @pytest.mark.parametrize("cmd,extra_args", [
        ('translate', ['--dx', '1', '--dy', '1']),
        ('rotate', ['--angle', '15', '--force']),
        ('scale', ['--factor', '1.5']),
        ('crop', ['--x', '1', '--y', '1', '--w', '5', '--h', '5']),
        ('inverse', []),
        ('log', []),
        ('gamma', ['--gamma', '2.0']),
        ('contrast', ['--intervals', '0,255', '--targets', '50,200']),
        ('creative', ['--threshold', '128']),
    ])
    def test_rgb_produces_valid_output(self, rgb_img, tmp_path, cmd, extra_args):
        inp, _ = rgb_img
        out = str(tmp_path / f"out_{cmd}.png")
        r = run_editor(cmd, '-i', inp, '-o', out, *extra_args)
        assert r.returncode == 0, f"stderr: {r.stderr}"
        result = imageio.imread(out)
        assert result.ndim == 3, f"Esperava RGB, obteve ndim={result.ndim}"
        assert result.shape[2] == 3


# ---------------------------------------------------------------------------
# Argumentos invalidos
# ---------------------------------------------------------------------------

class TestInvalidArgs:
    def test_missing_input(self, tmp_path):
        r = run_editor('inverse', '-o', str(tmp_path / "out.png"))
        assert r.returncode != 0

    def test_unknown_command(self):
        r = run_editor('foobar')
        assert r.returncode != 0

    def test_missing_required_flag(self, tmp_path, gray_img):
        inp, _ = gray_img
        out = str(tmp_path / "out.png")
        # translate sem --dx/--dy
        r = run_editor('translate', '-i', inp, '-o', out)
        assert r.returncode != 0
