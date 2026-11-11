import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from biocofre.config import Config
from biocofre.pipeline.caracteristicas import (N_UNIFORME, TABELA_UNIFORME, lbp_bruto, pesos_regioes,
                                               vetor_de_caracteristicas)
from biocofre.pipeline.classificacao import Candidato, Galeria, decidir, distancias
from biocofre.pipeline.executor import Pipeline
from biocofre.pipeline.preprocessamento import equalizar, medir_qualidade, para_cinza, reduzir_ruido
from conftest import rosto_sintetico


def test_lbp_codigo_conhecido():
    trecho = np.array([[6, 5, 2], [7, 6, 1], [9, 8, 7]], np.uint8)
    assert lbp_bruto(trecho)[0, 0] == 241          # bits: 1,0,0,0,1,1,1,1
    assert lbp_bruto(np.full((5, 5), 9, np.uint8)).min() == 255


def test_tabela_uniforme_tem_58_padroes_mais_um():
    assert set(np.unique(TABELA_UNIFORME)) == set(range(N_UNIFORME))
    assert (TABELA_UNIFORME == N_UNIFORME - 1).sum() == 256 - 58


def test_vetor_de_caracteristicas_dimensao_e_normalizacao():
    v = vetor_de_caracteristicas(rosto_sintetico(1), grade=8)
    assert v.shape == (8 * 8 * N_UNIFORME,)
    assert np.allclose(v.reshape(64, -1).sum(axis=1), 1.0, atol=1e-5)


def test_pesos_das_regioes_zeram_cantos_com_mascara():
    w = pesos_regioes(8, 100, True, 0.4, 0.5)
    assert w[0] == 0 and w[-1] == 0 and w[27] == 1.0
    assert np.all(pesos_regioes(8, 100, False) == 1.0)


def test_preprocessamento_formas_e_tipos():
    cor = np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8)
    cinza = para_cinza(cor)
    assert cinza.shape == (60, 80) and cinza.dtype == np.uint8
    assert reduzir_ruido(cinza, 3).shape == cinza.shape
    assert np.array_equal(reduzir_ruido(cinza, 0), cinza)
    assert np.array_equal(equalizar(cinza, 0), cinza)


def test_qualidade_rejeita_desfocada_escura_e_estourada(cfg):
    nitida = np.random.default_rng(2).integers(60, 190, (100, 100), dtype=np.uint8)
    assert medir_qualidade(nitida).avaliar(cfg) == (True, "ok")
    assert medir_qualidade(cv2.GaussianBlur(nitida, (0, 0), 6)).avaliar(cfg)[1] == "imagem_desfocada"
    assert medir_qualidade((nitida * 0.2).astype(np.uint8)).avaliar(cfg)[1] == "imagem_escura"
    assert medir_qualidade(np.clip(nitida.astype(int) + 150, 0, 255).astype(np.uint8)).avaliar(cfg)[1] == "imagem_estourada"


def test_distancia_qui_quadrado():
    g = np.random.default_rng(3).random((4, 64, 59)).astype(np.float32)
    g /= g.sum(axis=2, keepdims=True)
    w = np.ones(64, np.float32)
    d = distancias(g, g[2], w)
    assert d[2] == pytest.approx(0, abs=1e-6) and d.argmin() == 2 and np.all(d[[0, 1, 3]] > 0)


def test_decidir_regras():
    assert decidir([], 0.4).motivo == "galeria_vazia"
    assert decidir([Candidato(1, 0.5)], 0.4).motivo == "distancia_acima_do_limiar"
    assert decidir([Candidato(1, 0.30), Candidato(2, 0.32)], 0.4, 0.9).motivo == "ambiguo"
    ok = decidir([Candidato(1, 0.30), Candidato(2, 0.50)], 0.4, 0.9)
    assert ok.aceito and ok.usuario_id == 1
    assert decidir([Candidato(1, 0.3)], 0.4, 0.9).aceito   # um só cadastrado: sem razão


def test_identificacao_com_rostos_sinteticos(cfg):
    pipe = Pipeline(cfg)
    galeria = Galeria(cfg.lbp_grade, N_UNIFORME, pipe.pesos)
    for pessoa in range(1, 6):
        for v in range(3):
            galeria.adicionar(pessoa, pipe.extrair(rosto_sintetico(pessoa, v), ja_recortado=True).vetor)
    assert galeria.usuarios == [1, 2, 3, 4, 5]
    for pessoa in range(1, 6):
        res = pipe.processar(rosto_sintetico(pessoa, 9), galeria, ja_recortado=True)
        assert res.ranking[0].usuario_id == pessoa
        assert res.ranking[0].distancia < 0.7 * res.ranking[1].distancia
    desconhecido = pipe.processar(rosto_sintetico(77, 0), galeria, ja_recortado=True)
    assert desconhecido.ranking[0].distancia > 0.3


def test_pipeline_sem_rosto_em_imagem_uniforme(cfg):
    res = Pipeline(cfg).extrair(np.full((480, 640, 3), 90, np.uint8))
    assert res.problema == "sem_rosto" and res.vetor is None


def test_galeria_remover_e_adicionar_varios(cfg):
    pipe = Pipeline(cfg)
    g = Galeria(cfg.lbp_grade, N_UNIFORME, pipe.pesos)
    v = [pipe.extrair(rosto_sintetico(s), ja_recortado=True).vetor for s in (1, 2, 3)]
    g.adicionar_varios([1, 2, 3], v)
    g.remover(2)
    assert g.usuarios == [1, 3] and len(g) == 2


ORL = os.environ.get("BIOCOFRE_ORL")


@pytest.mark.skipif(not ORL or not Path(ORL).exists(), reason="defina BIOCOFRE_ORL com a pasta da base ORL")
def test_integracao_orl_detecta_e_identifica(cfg):
    from biocofre import avaliacao as av
    base = av.carregar_base(ORL, range(1, 11))
    vetores, detectado, pesos, _ = av.extrair_vetores(base, cfg, "quadro")
    assert detectado.mean() > 0.95
    D = av.matriz_distancias(vetores, cfg.lbp_grade, pesos)
    rank1, _ = av.identificacao_rank1(D, base.pessoas, detectado, 5, 10)
    assert rank1 > 0.90
