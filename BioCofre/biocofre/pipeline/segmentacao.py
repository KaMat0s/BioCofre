"""Fase 3 - Segmentação: detecção do rosto (Viola-Jones), recorte, normalização e máscara elíptica."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np

from .preprocessamento import ResultadoPre

Caixa = tuple[int, int, int, int]  # x, y, largura, altura


@dataclass
class ResultadoSeg:
    caixas: list[Caixa] = field(default_factory=list)
    caixa: Optional[Caixa] = None
    recorte: Optional[np.ndarray] = None         # equalizado, normalizado e mascarado
    recorte_cinza: Optional[np.ndarray] = None   # cinza normalizado (para medir qualidade)
    varios_rostos: bool = False


class DetectorFace:
    """Detector Haar em cascata (OpenCV). Devolve as caixas da maior para a menor."""

    def __init__(self, cfg) -> None:
        arquivo = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        self._cascata = cv2.CascadeClassifier(arquivo)
        if self._cascata.empty():
            raise RuntimeError(f"Não foi possível carregar a cascata Haar em '{arquivo}'.")
        self._escala = cfg.haar_escala
        self._vizinhos = cfg.haar_vizinhos
        self._minimo = cfg.face_min_px

    def _varrer(self, imagem: np.ndarray) -> list[Caixa]:
        caixas = self._cascata.detectMultiScale(
            imagem, scaleFactor=self._escala, minNeighbors=self._vizinhos,
            minSize=(self._minimo, self._minimo),
        )
        return [tuple(int(v) for v in c) for c in caixas]

    def detectar(self, *imagens: np.ndarray) -> list[Caixa]:
        """Tenta cada imagem em ordem e devolve as caixas da primeira que achar algum rosto."""
        for imagem in imagens:
            caixas = self._varrer(imagem)
            if caixas:
                return sorted(caixas, key=lambda c: c[2] * c[3], reverse=True)
        return []


@lru_cache(maxsize=16)
def mascara_eliptica(tam: int, rx: float = 0.50, ry: float = 0.50) -> np.ndarray:
    """Elipse centralizada que preserva o rosto e zera os cantos (fundo)."""
    mascara = np.zeros((tam, tam), np.uint8)
    cv2.ellipse(mascara, (tam // 2, tam // 2), (int(tam * rx), int(tam * ry)),
                0, 0, 360, 255, -1)
    return mascara


def _redimensionar(imagem: np.ndarray, tam: int) -> np.ndarray:
    reduzindo = imagem.shape[0] > tam or imagem.shape[1] > tam
    interp = cv2.INTER_AREA if reduzindo else cv2.INTER_CUBIC
    return cv2.resize(imagem, (tam, tam), interpolation=interp)


def _mascarar(imagem: np.ndarray, cfg) -> np.ndarray:
    if not cfg.mascara_eliptica:
        return imagem
    mascara = mascara_eliptica(cfg.tam_face, cfg.mascara_rx, cfg.mascara_ry)
    return cv2.bitwise_and(imagem, imagem, mask=mascara)


def recortar_face(imagem: np.ndarray, caixa: Caixa, tam: int, margem: float) -> np.ndarray:
    """Recorta a caixa com folga proporcional e redimensiona para ``tam`` x ``tam``."""
    x, y, w, h = caixa
    mx, my = int(w * margem), int(h * margem)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(imagem.shape[1], x + w + mx), min(imagem.shape[0], y + h + my)
    return _redimensionar(imagem[y0:y1, x0:x1], tam)


def segmentar(pre: ResultadoPre, cfg, detector: Optional[DetectorFace],
              ja_recortado: bool = False) -> ResultadoSeg:
    """Isola o rosto. Com ``ja_recortado`` a imagem inteira é tratada como o rosto (bases de teste)."""
    tam = cfg.tam_face
    fonte = pre.equalizada if cfg.equalizar_caracteristicas else pre.suavizada
    if ja_recortado:
        h, w = fonte.shape[:2]
        caixa = (0, 0, w, h)
        rec = _mascarar(_redimensionar(fonte, tam), cfg)
        return ResultadoSeg([caixa], caixa, rec, _redimensionar(pre.cinza, tam), False)

    # A cascata Haar é mais confiável sem equalização; a versão equalizada fica como reserva.
    # (O recorte usado no descritor vem de ``fonte``, definida pela configuração.)
    caixas = detector.detectar(pre.suavizada, pre.equalizada)
    if not caixas:
        return ResultadoSeg()
    principal = caixas[0]
    area0 = principal[2] * principal[3]
    varios = any(c[2] * c[3] >= 0.5 * area0 for c in caixas[1:])
    rec = _mascarar(recortar_face(fonte, principal, tam, cfg.margem_face), cfg)
    rec_cinza = recortar_face(pre.cinza, principal, tam, cfg.margem_face)
    return ResultadoSeg(caixas, principal, rec, rec_cinza, varios)
