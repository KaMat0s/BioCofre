"""Fase 5 - Classificação e interpretação: vizinho mais próximo com distância qui-quadrado.

A galeria guarda os modelos (vetores) de cada usuário. A classificação devolve um ranking
de usuários, com a menor distância entre o vetor consultado e os modelos de cada um. A
interpretação (``decidir``) aplica o limiar de distância e a razão entre o primeiro e o
segundo colocados; a política de acesso por nível fica em ``biocofre.acesso``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

EPS = 1e-10


def distancias(galeria: np.ndarray, consulta: np.ndarray, pesos: np.ndarray,
               metrica: str = "chi2") -> np.ndarray:
    """Distância entre ``consulta`` (R, B) e cada modelo de ``galeria`` (N, R, B).

    O resultado é a média ponderada, por região, da distância entre os histogramas.
    """
    if galeria.shape[0] == 0:
        return np.empty(0, np.float32)
    if metrica == "chi2":
        por_regiao = (((galeria - consulta) ** 2) / (galeria + consulta + EPS)).sum(axis=2)
    elif metrica == "euclidiana":
        por_regiao = np.sqrt(((galeria - consulta) ** 2).sum(axis=2))
    elif metrica == "intersecao":
        por_regiao = 1.0 - np.minimum(galeria, consulta).sum(axis=2)
    else:
        raise ValueError(f"Métrica desconhecida: {metrica}")
    return (por_regiao * pesos).sum(axis=1) / pesos.sum()


@dataclass
class Candidato:
    usuario_id: int
    distancia: float


@dataclass
class Decisao:
    usuario_id: Optional[int]
    distancia: float
    segunda: Optional[float]
    aceito: bool
    motivo: str   # ok | galeria_vazia | distancia_acima_do_limiar | ambiguo


class Galeria:
    """Conjunto de modelos biométricos em memória."""

    def __init__(self, grade: int, n_categorias: int, pesos: np.ndarray, metrica: str = "chi2") -> None:
        self._regioes = grade * grade
        self._categorias = n_categorias
        self._pesos = pesos.astype(np.float32)
        self._metrica = metrica
        self._vetores = np.empty((0, self._regioes, n_categorias), np.float32)
        self._rotulos = np.empty(0, np.int64)

    def __len__(self) -> int:
        return int(self._rotulos.size)

    @property
    def usuarios(self) -> list[int]:
        return sorted(int(u) for u in np.unique(self._rotulos))

    def adicionar(self, usuario_id: int, vetor: np.ndarray) -> None:
        v = np.asarray(vetor, np.float32).reshape(1, self._regioes, self._categorias)
        self._vetores = np.concatenate([self._vetores, v])
        self._rotulos = np.append(self._rotulos, usuario_id)

    def adicionar_varios(self, usuario_ids, vetores) -> None:
        v = np.asarray(vetores, np.float32).reshape(-1, self._regioes, self._categorias)
        self._vetores = np.concatenate([self._vetores, v])
        self._rotulos = np.concatenate([self._rotulos, np.asarray(usuario_ids, np.int64)])

    def remover(self, usuario_id: int) -> None:
        manter = self._rotulos != usuario_id
        self._vetores, self._rotulos = self._vetores[manter], self._rotulos[manter]

    def ranking(self, vetor: np.ndarray) -> list[Candidato]:
        """Um candidato por usuário, do mais próximo ao mais distante."""
        if len(self) == 0:
            return []
        consulta = np.asarray(vetor, np.float32).reshape(self._regioes, self._categorias)
        d = distancias(self._vetores, consulta, self._pesos, self._metrica)
        lista = [Candidato(int(u), float(d[self._rotulos == u].min())) for u in np.unique(self._rotulos)]
        return sorted(lista, key=lambda c: c.distancia)


def decidir(ranking: list[Candidato], limiar: float, razao_max: float = 1.0) -> Decisao:
    """Aplica o limiar de distância e a razão d1/d2 ao melhor candidato do ranking."""
    if not ranking:
        return Decisao(None, float("inf"), None, False, "galeria_vazia")
    primeiro = ranking[0]
    segunda = ranking[1].distancia if len(ranking) > 1 else None
    if primeiro.distancia > limiar:
        return Decisao(primeiro.usuario_id, primeiro.distancia, segunda, False,
                       "distancia_acima_do_limiar")
    if segunda is not None and segunda > 0 and primeiro.distancia / segunda > razao_max:
        return Decisao(primeiro.usuario_id, primeiro.distancia, segunda, False, "ambiguo")
    return Decisao(primeiro.usuario_id, primeiro.distancia, segunda, True, "ok")
