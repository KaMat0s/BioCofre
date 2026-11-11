"""Fase 4 - Extração de características: LBP uniforme em histogramas por região.

O rosto normalizado é convertido em um mapa de códigos LBP (comparação de cada pixel com
seus 8 vizinhos). O mapa é dividido em uma grade de regiões e o histograma de cada região
(59 categorias, no modo uniforme) é concatenado em um único vetor de características.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .segmentacao import mascara_eliptica

N_UNIFORME = 59   # 58 padrões uniformes + 1 categoria para os demais
N_BRUTO = 256

# Vizinhos no sentido horário, a partir do canto superior esquerdo.
_VIZINHOS = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))


def _construir_tabela_uniforme() -> np.ndarray:
    tabela = np.full(256, N_UNIFORME - 1, dtype=np.uint8)
    proximo = 0
    for valor in range(256):
        bits = [(valor >> i) & 1 for i in range(8)]
        transicoes = sum(bits[i] != bits[(i + 1) % 8] for i in range(8))
        if transicoes <= 2:
            tabela[valor] = proximo
            proximo += 1
    assert proximo == N_UNIFORME - 1
    return tabela


TABELA_UNIFORME = _construir_tabela_uniforme()


def lbp_bruto(cinza: np.ndarray) -> np.ndarray:
    """Código LBP (P=8, R=1) de cada pixel interno; o mapa tem 2 linhas e 2 colunas a menos."""
    g = cinza.astype(np.int16)
    altura, largura = g.shape
    centro = g[1:-1, 1:-1]
    codigo = np.zeros((altura - 2, largura - 2), np.uint8)
    for bit, (dy, dx) in enumerate(_VIZINHOS):
        vizinho = g[1 + dy: altura - 1 + dy, 1 + dx: largura - 1 + dx]
        codigo |= (vizinho >= centro).astype(np.uint8) << bit
    return codigo


@lru_cache(maxsize=16)
def _mapa_regioes(altura: int, largura: int, grade: int) -> np.ndarray:
    linhas = np.minimum((np.arange(altura) * grade) // altura, grade - 1)
    colunas = np.minimum((np.arange(largura) * grade) // largura, grade - 1)
    return (linhas[:, None] * grade + colunas[None, :]).astype(np.int32)


def vetor_de_caracteristicas(recorte: np.ndarray, grade: int = 8, uniforme: bool = True) -> np.ndarray:
    """Vetor de ``grade*grade*categorias`` valores (histogramas normalizados por região)."""
    codigos = lbp_bruto(recorte)
    rotulos = TABELA_UNIFORME[codigos] if uniforme else codigos
    n = N_UNIFORME if uniforme else N_BRUTO
    mapa = _mapa_regioes(codigos.shape[0], codigos.shape[1], grade)
    contagem = np.bincount((mapa * n + rotulos).ravel(), minlength=grade * grade * n)
    contagem = contagem.reshape(grade * grade, n).astype(np.float32)
    total = contagem.sum(axis=1, keepdims=True)
    total[total == 0] = 1.0
    return (contagem / total).ravel()


def pesos_regioes(grade: int, tam: int, usar_mascara: bool,
                  rx: float = 0.50, ry: float = 0.50) -> np.ndarray:
    """Peso de cada região = fração de seus pixels que está dentro da máscara elíptica."""
    if not usar_mascara:
        return np.ones(grade * grade, np.float32)
    dentro = (mascara_eliptica(tam, rx, ry)[1:-1, 1:-1] > 0).astype(np.float32)
    mapa = _mapa_regioes(dentro.shape[0], dentro.shape[1], grade)
    soma = np.bincount(mapa.ravel(), weights=dentro.ravel(), minlength=grade * grade)
    cont = np.bincount(mapa.ravel(), minlength=grade * grade)
    return (soma / np.maximum(cont, 1)).astype(np.float32)


def imagem_lbp(recorte: np.ndarray) -> np.ndarray:
    """Mapa de códigos LBP em tons de cinza (0-255), para visualização na interface."""
    return lbp_bruto(recorte)
