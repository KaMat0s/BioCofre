"""Fase 1 - Aquisição: obtém quadros de uma câmera, de um vídeo ou de arquivos de imagem."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Protocol

import cv2
import numpy as np

EXTENSOES = {".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


class ErroAquisicao(RuntimeError):
    """Falha ao abrir uma fonte de imagens ou ao ler uma imagem."""


class FonteImagem(Protocol):
    def ler(self) -> Optional[np.ndarray]: ...
    def fechar(self) -> None: ...


def carregar_imagem(caminho: str | Path) -> np.ndarray:
    """Lê uma imagem colorida (BGR). Usa ``imdecode`` para aceitar caminhos com acentos no Windows."""
    caminho = Path(caminho)
    try:
        dados = np.fromfile(str(caminho), dtype=np.uint8)
    except OSError as exc:
        raise ErroAquisicao(f"Não foi possível ler o arquivo '{caminho}': {exc}") from exc
    imagem = cv2.imdecode(dados, cv2.IMREAD_COLOR) if dados.size else None
    if imagem is None:
        raise ErroAquisicao(f"O arquivo '{caminho}' não é uma imagem válida.")
    return imagem


def listar_imagens(pasta: str | Path) -> list[Path]:
    """Lista, em ordem, os arquivos de imagem de uma pasta (inclui subpastas)."""
    arquivos = [p for p in Path(pasta).rglob("*") if p.suffix.lower() in EXTENSOES]
    return sorted(arquivos)


class _CapturaOpenCV:
    def __init__(self, origem, espelhar: bool, api: int = cv2.CAP_ANY) -> None:
        self._cap = cv2.VideoCapture(origem, api)
        self._espelhar = espelhar

    def ler(self) -> Optional[np.ndarray]:
        ok, quadro = self._cap.read()
        if not ok or quadro is None:
            return None
        return cv2.flip(quadro, 1) if self._espelhar else quadro

    def fechar(self) -> None:
        self._cap.release()


class Camera(_CapturaOpenCV):
    """Webcam acessada pelo OpenCV."""

    def __init__(self, indice: int = 0, largura: int = 640, altura: int = 480,
                 espelhar: bool = False) -> None:
        api = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        super().__init__(indice, espelhar, api)
        if not self._cap.isOpened():
            self._cap.release()
            raise ErroAquisicao(
                f"Não foi possível abrir a câmera {indice}. Verifique se ela está conectada "
                "e se nenhum outro programa a está usando."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)


class VideoArquivo(_CapturaOpenCV):
    """Arquivo de vídeo (útil para demonstrações e testes sem câmera)."""

    def __init__(self, caminho: str | Path) -> None:
        super().__init__(str(caminho), False)
        if not self._cap.isOpened():
            self._cap.release()
            raise ErroAquisicao(f"Não foi possível abrir o vídeo '{caminho}'.")


class SequenciaDeArquivos:
    """Reproduz uma lista de imagens como se fosse uma câmera.

    Com ``fps`` os quadros avançam pelo relógio (como um vídeo) e ``ler`` devolve ``None`` enquanto
    o próximo quadro ainda não chegou; sem ``fps`` cada chamada devolve a imagem seguinte.
    """

    def __init__(self, caminhos: Iterable[str | Path], repetir: bool = True,
                 fps: Optional[float] = None) -> None:
        self._caminhos = [Path(c) for c in caminhos]
        if not self._caminhos:
            raise ErroAquisicao("Nenhuma imagem encontrada na origem informada.")
        self._i = 0
        self._repetir = repetir
        self._fps = fps
        self._t0: Optional[float] = None
        self._ultimo = -1
        self._cache: dict[Path, np.ndarray] = {}

    def _carregar(self, caminho: Path) -> np.ndarray:
        if caminho not in self._cache:
            if len(self._cache) > 200:
                self._cache.clear()
            self._cache[caminho] = carregar_imagem(caminho)
        return self._cache[caminho].copy()

    def ler(self) -> Optional[np.ndarray]:
        n = len(self._caminhos)
        if self._fps:
            if self._t0 is None:
                self._t0 = time.monotonic()
            i = int((time.monotonic() - self._t0) * self._fps)
            if i >= n:
                if not self._repetir:
                    return None
                i %= n
            if i == self._ultimo:
                return None
            self._ultimo = i
            return self._carregar(self._caminhos[i])
        if self._i >= n:
            if not self._repetir:
                return None
            self._i = 0
        imagem = self._carregar(self._caminhos[self._i])
        self._i += 1
        return imagem

    def fechar(self) -> None:
        self._cache.clear()


def _ler_roteiro(pasta: Path) -> tuple[list[Path], float]:
    """Lê ``roteiro.txt``: um nome de arquivo por linha (pode repetir); ``# fps=15`` define a taxa."""
    fps, caminhos = 15.0, []
    for linha in (pasta / "roteiro.txt").read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if linha.startswith("# fps="):
            fps = float(linha.split("=")[1])
        elif linha and not linha.startswith("#"):
            caminhos.append(pasta / linha)
    return caminhos, fps


def abrir_fonte(origem, cfg) -> FonteImagem:
    """Abre a fonte conforme ``origem``: índice de câmera, pasta, imagem ou vídeo."""
    if isinstance(origem, int) or str(origem).isdigit():
        return Camera(int(origem), cfg.largura, cfg.altura, cfg.espelhar)
    caminho = Path(str(origem))
    if caminho.is_dir():
        if (caminho / "roteiro.txt").exists():
            roteiro, fps = _ler_roteiro(caminho)
            return SequenciaDeArquivos(roteiro, fps=fps)
        return SequenciaDeArquivos(listar_imagens(caminho))
    if caminho.suffix.lower() in EXTENSOES and caminho.is_file():
        return SequenciaDeArquivos([caminho])
    if caminho.is_file():
        return VideoArquivo(caminho)
    raise ErroAquisicao(f"Origem de imagens inválida: '{origem}'.")
