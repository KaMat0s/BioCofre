"""Fase 2 - Pré-processamento: tons de cinza, redução de ruído e equalização adaptativa."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ResultadoPre:
    cinza: np.ndarray
    suavizada: np.ndarray
    equalizada: np.ndarray


@dataclass
class Qualidade:
    nitidez: float      # variância do laplaciano (quanto maior, mais nítido)
    brilho: float       # média dos níveis de cinza
    contraste: float    # desvio padrão dos níveis de cinza

    def avaliar(self, cfg) -> tuple[bool, str]:
        """Devolve (aprovada, motivo)."""
        if self.nitidez < cfg.nitidez_min:
            return False, "imagem_desfocada"
        if self.brilho < cfg.brilho_min:
            return False, "imagem_escura"
        if self.brilho > cfg.brilho_max:
            return False, "imagem_estourada"
        if self.contraste < cfg.contraste_min:
            return False, "baixo_contraste"
        return True, "ok"


def para_cinza(imagem: np.ndarray) -> np.ndarray:
    """Converte para tons de cinza: Y = 0,299 R + 0,587 G + 0,114 B."""
    if imagem.ndim == 2:
        return imagem.copy()
    if imagem.shape[2] == 4:
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)


def reduzir_ruido(cinza: np.ndarray, kernel: int = 3) -> np.ndarray:
    """Filtro gaussiano; ``kernel`` <= 1 desliga a etapa."""
    if kernel <= 1:
        return cinza.copy()
    k = kernel if kernel % 2 == 1 else kernel + 1
    return cv2.GaussianBlur(cinza, (k, k), 0)


def equalizar(cinza: np.ndarray, clip: float = 2.0, grade: int = 8) -> np.ndarray:
    """Equalização adaptativa com limite de contraste (CLAHE); ``clip`` <= 0 desliga."""
    if clip <= 0:
        return cinza.copy()
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grade, grade)).apply(cinza)


def preprocessar(imagem: np.ndarray, cfg) -> ResultadoPre:
    cinza = para_cinza(imagem)
    suavizada = reduzir_ruido(cinza, cfg.gauss_kernel)
    equalizada = equalizar(suavizada, cfg.clahe_clip, cfg.clahe_grade)
    return ResultadoPre(cinza=cinza, suavizada=suavizada, equalizada=equalizada)


def medir_qualidade(face_cinza: np.ndarray) -> Qualidade:
    """Mede nitidez, brilho e contraste do recorte do rosto (antes da equalização)."""
    nitidez = float(cv2.Laplacian(face_cinza, cv2.CV_64F).var())
    return Qualidade(nitidez=nitidez, brilho=float(face_cinza.mean()),
                     contraste=float(face_cinza.std()))
