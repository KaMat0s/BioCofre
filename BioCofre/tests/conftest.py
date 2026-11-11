import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biocofre.banco import Banco  # noqa: E402
from biocofre.config import Config  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    return Config(pasta_dados=str(tmp_path / "dados"), resultado_exibido_s=1.0, rearme_sem_rosto_s=0.5)


@pytest.fixture
def banco(cfg):
    b = Banco(cfg.caminho_banco, cfg.caminho_chave, iteracoes_kdf=1000)
    yield b
    b.fechar()


def rosto_sintetico(semente: int, variante: int = 0, tam: int = 100) -> np.ndarray:
    """'Pessoa' sintética: textura suave própria da semente; as variantes só mudam ruído e brilho."""
    base = np.random.default_rng(semente).random((tam, tam)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), 2.2)
    base = (base - base.min()) / (base.max() - base.min())
    rng = np.random.default_rng(1000 * semente + variante + 1)
    img = base * (0.8 + 0.1 * rng.random()) + 0.1 * rng.random() + 0.02 * rng.standard_normal(base.shape)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)
