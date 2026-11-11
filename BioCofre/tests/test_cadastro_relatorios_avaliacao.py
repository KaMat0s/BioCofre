import numpy as np
import pytest

from biocofre import avaliacao as av
from biocofre import relatorios
from biocofre.cadastro import Amostra, Cadastro, ErroCadastro
from biocofre.acesso import montar_galeria
from biocofre.pipeline.caracteristicas import N_UNIFORME
from biocofre.pipeline.executor import Pipeline
from conftest import rosto_sintetico


def _amostras(pipe, semente, n=6):
    return [Amostra(True, "ok", pipe.extrair(rosto_sintetico(semente, v), ja_recortado=True).vetor)
            for v in range(n)]


@pytest.fixture
def cad(cfg, banco):
    pipe = Pipeline(cfg)
    return Cadastro(banco, pipe, cfg), pipe


def test_cadastro_valida_entradas(cad):
    c, pipe = cad
    am = _amostras(pipe, 1)
    from biocofre.cadastro import Amostra as A
    am = [A(True, "ok", a.vetor, type("Q", (), {"nitidez": 100.0})()) for a in am]
    with pytest.raises(ErroCadastro, match="nome"):
        c.cadastrar("  ", "", 1, None, None, am)
    with pytest.raises(ErroCadastro, match="divisão"):
        c.cadastrar("Ana", "", 2, None, None, am)
    with pytest.raises(ErroCadastro, match="PIN"):
        c.cadastrar("Min", "", 3, None, None, am)
    with pytest.raises(ErroCadastro, match="PIN"):
        c.cadastrar("Min", "", 3, None, "12", am)
    with pytest.raises(ErroCadastro, match="amostras"):
        c.cadastrar("Ana", "", 1, None, None, am[:3])


def test_cadastro_grava_modelos_e_registra_no_log(cad, banco):
    c, pipe = cad
    am = [Amostra(True, "ok", a.vetor, type("Q", (), {"nitidez": 100.0})()) for a in _amostras(pipe, 1)]
    u = c.cadastrar("Ana", "Analista", 1, None, None, am)
    assert len(banco.carregar_modelos()) == 6
    assert banco.listar_acessos()[0]["evento"] == "CADASTRO" and banco.verificar_integridade().ok


def test_cadastro_recusa_rosto_duplicado(cad, banco):
    c, pipe = cad
    q = type("Q", (), {"nitidez": 100.0})()
    c.cadastrar("Ana", "", 1, None, None, [Amostra(True, "ok", a.vetor, q) for a in _amostras(pipe, 1)])
    galeria = montar_galeria(banco, pipe)
    de_novo = [Amostra(True, "ok", pipe.extrair(rosto_sintetico(1, 50 + v), ja_recortado=True).vetor, q) for v in range(6)]
    with pytest.raises(ErroCadastro, match="parecido"):
        c.cadastrar("Ana 2", "", 1, None, None, de_novo, galeria)
    outra = [Amostra(True, "ok", a.vetor, q) for a in _amostras(pipe, 2)]
    assert c.cadastrar("Beto", "", 1, None, None, outra, galeria).nome == "Beto"


def test_avaliar_quadro_rejeita_sem_rosto(cad):
    c, _ = cad
    a = c.avaliar_quadro(np.full((480, 640, 3), 90, np.uint8))
    assert not a.aceita and a.motivo == "sem_rosto"


def test_relatorios(banco, tmp_path):
    banco.registrar_acesso("CONCEDIDO", 1, None, "Ana", "ok", 0.25)
    banco.registrar_acesso("NEGADO", 4, None, None, "face_nao_reconhecida", 0.5)
    registros = banco.listar_acessos(limite=0)
    r = relatorios.resumir(registros)
    assert r["total"] == 2 and r["por_evento"] == {"NEGADO": 1, "CONCEDIDO": 1}
    csv_ = relatorios.exportar_csv(registros, tmp_path / "a.csv")
    assert csv_.read_text(encoding="utf-8-sig").splitlines()[0].startswith("id;ts;zona")
    html_ = relatorios.gerar_html(banco, tmp_path / "r.html").read_text(encoding="utf-8")
    assert "Cadeia íntegra" in html_ and "face_nao_reconhecida" in html_


def test_taxas_eer_e_limiar():
    gen = np.linspace(0.20, 0.40, 200)
    imp = np.linspace(0.35, 0.70, 400)
    far, frr = av.taxas_erro(gen, imp, np.array([0.30, 0.45]))
    assert far[0] == 0 and frr[0] == pytest.approx(0.5, abs=0.02) and frr[1] == 0 and far[1] > 0.2
    e, tau = av.eer(gen, imp)
    assert 0 < e < 0.2 and 0.34 < tau < 0.40
    t = av.limiar_para_far(imp, 0.0)
    assert (imp <= t).sum() == 0 and av.limiar_para_far(imp, 0.01) > t


def test_calibrar_limiares_sao_progressivamente_mais_rigidos():
    imp = np.random.default_rng(0).normal(0.5, 0.05, 5000)
    lim = av.calibrar_limiares(imp)
    assert lim[1] > lim[2] > lim[3] and lim[2] <= lim[1] - 0.01 + 1e-9


def test_conjunto_aberto_com_separacao_perfeita():
    pessoas = np.repeat(np.arange(8), 10)
    D = np.where(pessoas[:, None] == pessoas[None, :], 0.2, 0.8).astype(np.float32)
    np.fill_diagonal(D, 0.0)
    det = np.ones(80, bool)
    r = av.conjunto_aberto(D, pessoas, det, limiar=0.4, razao_max=0.9, n_treino=5, n_intrusos=2, reps=10)
    assert r["acesso_correto"] == 1.0 and r["intruso_aceito"] == 0.0 and r["troca_de_identidade"] == 0.0
    d_int = av.distancias_intrusos(D, pessoas, det, 0.9, 5, 2, 10)
    assert np.all(d_int > 0.7)
    assert av.identificacao_rank1(D, pessoas, det, 5, 5)[0] == 1.0


def test_faixa():
    assert av.faixa("1-3,7") == [1, 2, 3, 7]
