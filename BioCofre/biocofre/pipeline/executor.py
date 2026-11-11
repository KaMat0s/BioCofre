"""Orquestra as fases 2 a 5 e registra o resultado intermediário e o tempo de cada fase."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .caracteristicas import imagem_lbp, pesos_regioes, vetor_de_caracteristicas
from .classificacao import Candidato, Galeria
from .preprocessamento import ResultadoPre, preprocessar
from .segmentacao import DetectorFace, ResultadoSeg, segmentar


@dataclass
class ResultadoPipeline:
    quadro: np.ndarray
    pre: Optional[ResultadoPre] = None
    seg: Optional[ResultadoSeg] = None
    lbp: Optional[np.ndarray] = None
    vetor: Optional[np.ndarray] = None
    ranking: list[Candidato] = field(default_factory=list)
    tempos_ms: dict[str, float] = field(default_factory=dict)
    problema: Optional[str] = None   # sem_rosto | varios_rostos


class Pipeline:
    """Fases 2 a 5 do processamento de imagens aplicadas a um quadro."""

    def __init__(self, cfg, detector: Optional[DetectorFace] = None) -> None:
        self.cfg = cfg
        self.detector = detector
        self.pesos = pesos_regioes(cfg.lbp_grade, cfg.tam_face, cfg.mascara_eliptica,
                                   cfg.mascara_rx, cfg.mascara_ry)

    def _detector(self) -> DetectorFace:
        if self.detector is None:
            self.detector = DetectorFace(self.cfg)
        return self.detector

    def extrair(self, quadro: np.ndarray, ja_recortado: bool = False) -> ResultadoPipeline:
        """Fases 2, 3 e 4: devolve o vetor de características do rosto, se houver um."""
        res = ResultadoPipeline(quadro=quadro)
        marcas = [time.perf_counter()]

        res.pre = preprocessar(quadro, self.cfg)
        marcas.append(time.perf_counter())

        detector = None if ja_recortado else self._detector()
        res.seg = segmentar(res.pre, self.cfg, detector, ja_recortado)
        marcas.append(time.perf_counter())

        if res.seg.recorte is None:
            res.problema = "sem_rosto"
        elif res.seg.varios_rostos:
            res.problema = "varios_rostos"
        else:
            res.lbp = imagem_lbp(res.seg.recorte)
            res.vetor = vetor_de_caracteristicas(res.seg.recorte, self.cfg.lbp_grade)
        marcas.append(time.perf_counter())

        for nome, ini, fim in zip(("preprocessamento", "segmentacao", "caracteristicas"),
                                  marcas, marcas[1:]):
            res.tempos_ms[nome] = (fim - ini) * 1000.0
        return res

    def classificar(self, res: ResultadoPipeline, galeria: Galeria) -> ResultadoPipeline:
        """Fase 5 (ranking de distâncias); a decisão final fica a cargo da política de acesso."""
        if res.vetor is not None:
            inicio = time.perf_counter()
            res.ranking = galeria.ranking(res.vetor)
            res.tempos_ms["classificacao"] = (time.perf_counter() - inicio) * 1000.0
        return res

    def processar(self, quadro: np.ndarray, galeria: Galeria,
                  ja_recortado: bool = False) -> ResultadoPipeline:
        return self.classificar(self.extrair(quadro, ja_recortado), galeria)
