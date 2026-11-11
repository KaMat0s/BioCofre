import numpy as np
import pytest

from biocofre.acesso import Autenticador, Estado, autorizado
from biocofre.banco import Usuario, Zona
from biocofre.pipeline.classificacao import Candidato
from biocofre.pipeline.executor import ResultadoPipeline


def _u(nivel, divisao=None, ativo=True):
    return Usuario(1, "X", "", nivel, divisao, ativo, False)


@pytest.mark.parametrize("usuario,zona,esperado", [
    (_u(1), Zona(1, "geral", 1, None), True),
    (_u(1), Zona(2, "div", 2, "Toxicologia"), False),
    (_u(2, "Toxicologia"), Zona(2, "div", 2, "Toxicologia"), True),
    (_u(2, "Biossegurança"), Zona(2, "div", 2, "Toxicologia"), False),
    (_u(2, "Toxicologia"), Zona(1, "geral", 1, None), True),
    (_u(2, "Toxicologia"), Zona(3, "cofre", 3, None), False),
    (_u(3), Zona(3, "cofre", 3, None), True),
    (_u(3), Zona(2, "div", 2, "Toxicologia"), True),
    (_u(3, ativo=False), Zona(1, "geral", 1, None), False),
])
def test_regra_dos_tres_niveis(usuario, zona, esperado):
    assert autorizado(usuario, zona)[0] is esperado


class PipelineFalso:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pesos = np.ones(cfg.lbp_grade ** 2, np.float32)
        self.proximo = None

    def processar(self, quadro, galeria, ja_recortado=False):
        return self.proximo


class Relogio:
    t = 100.0

    def __call__(self):
        return self.t


def quadro(uid=None, dist=0.2, segunda=0.8, problema=None):
    ranking = [] if uid is None else [Candidato(uid, dist), Candidato(999, segunda)]
    return ResultadoPipeline(quadro=None, ranking=ranking, problema=problema)


@pytest.fixture
def ambiente(cfg, banco):
    pipe, rel = PipelineFalso(cfg), Relogio()
    auth = Autenticador(banco, cfg, pipe, rel)
    zonas = {z.nivel: z for z in banco.zonas()}
    return auth, pipe, rel, zonas, banco


def alimentar(auth, pipe, resultado, n=1):
    for _ in range(n):
        pipe.proximo = resultado
        sit, _ = auth.processar(None)
    return sit


def liberar_terminal(auth, pipe, rel):
    """Passa o tempo de exibição e tira o rosto do quadro para rearmar o terminal."""
    rel.t += 1.2
    pipe.proximo = quadro(problema="sem_rosto")
    auth.processar(None)
    rel.t += 0.6
    auth.processar(None)


def test_concede_apos_quadros_consecutivos(ambiente):
    auth, pipe, rel, zonas, banco = ambiente
    u = banco.criar_usuario("Ana", "", 1, None, None, [])
    auth.definir_zona(zonas[1])
    assert alimentar(auth, pipe, quadro(u.id), 2).estado == Estado.ANALISANDO
    sit = alimentar(auth, pipe, quadro(u.id))
    assert sit.estado == Estado.CONCEDIDO and sit.usuario.id == u.id
    log = banco.listar_acessos()
    assert log[0]["evento"] == "CONCEDIDO" and log[0]["usuario_nome"] == "Ana"


def test_identidades_diferentes_nao_confirmam(ambiente):
    auth, pipe, rel, zonas, banco = ambiente
    a = banco.criar_usuario("Ana", "", 1, None, None, [])
    b = banco.criar_usuario("Beto", "", 1, None, None, [])
    auth.definir_zona(zonas[1])
    for i in range(6):
        sit = alimentar(auth, pipe, quadro(a.id if i % 2 == 0 else b.id))
    assert sit.estado == Estado.ANALISANDO and banco.listar_acessos() == []


def test_distancia_acima_do_limiar_nao_concede(ambiente):
    auth, pipe, rel, zonas, banco = ambiente
    u = banco.criar_usuario("Ana", "", 1, None, None, [])
    auth.definir_zona(zonas[3])
    sit = alimentar(auth, pipe, quadro(u.id, dist=0.40), 20)   # limiar do Nível 3 é 0,386
    assert sit.estado == Estado.NEGADO and "não reconhecido" in sit.mensagem


def test_nivel_insuficiente_e_divisao_errada(ambiente):
    auth, pipe, rel, zonas, banco = ambiente
    servidor = banco.criar_usuario("Ana", "", 1, None, None, [])
    auth.definir_zona(zonas[2])
    sit = alimentar(auth, pipe, quadro(servidor.id), 5)
    assert sit.estado == Estado.NEGADO and banco.listar_acessos()[0]["motivo"] == "nao_autorizado:nivel_insuficiente"
    liberar_terminal(auth, pipe, rel)
    diretor = banco.criar_usuario("Dora", "", 2, "Biossegurança", None, [])
    zona_tox = next(z for z in banco.zonas() if z.divisao == "Toxicologia")
    auth.definir_zona(zona_tox)
    sit = alimentar(auth, pipe, quadro(diretor.id), 5)
    assert sit.estado == Estado.NEGADO and banco.listar_acessos()[0]["motivo"] == "nao_autorizado:divisao_diferente"


def test_nivel3_exige_pin(ambiente, cfg):
    auth, pipe, rel, zonas, banco = ambiente
    m = banco.criar_usuario("Ministro", "", 3, None, "1234", [])
    auth.definir_zona(zonas[3])
    sit = alimentar(auth, pipe, quadro(m.id, dist=0.2), 8)
    assert sit.estado == Estado.PIN_NECESSARIO and banco.listar_acessos() == []
    assert auth.informar_pin("0000").estado == Estado.PIN_NECESSARIO
    assert auth.informar_pin("1111").estado == Estado.PIN_NECESSARIO
    assert auth.informar_pin("2222").estado == Estado.NEGADO
    assert banco.listar_acessos()[0]["motivo"] == "pin_incorreto"


def test_nivel3_com_pin_correto_concede(ambiente):
    auth, pipe, rel, zonas, banco = ambiente
    m = banco.criar_usuario("Ministro", "", 3, None, "1234", [])
    auth.definir_zona(zonas[3])
    alimentar(auth, pipe, quadro(m.id, dist=0.2), 8)
    assert auth.informar_pin("1234").estado == Estado.CONCEDIDO


def test_pin_expira_e_cancelar_conta_como_negado(ambiente):
    auth, pipe, rel, zonas, banco = ambiente
    m = banco.criar_usuario("Ministro", "", 3, None, "1234", [])
    auth.definir_zona(zonas[3])
    alimentar(auth, pipe, quadro(m.id, dist=0.2), 8)
    assert auth.cancelar_pin().estado == Estado.NEGADO
    assert banco.listar_acessos()[0]["motivo"] == "pin_cancelado"
    liberar_terminal(auth, pipe, rel)
    alimentar(auth, pipe, quadro(m.id, dist=0.2), 8)
    rel.t += 31
    pipe.proximo = quadro(m.id)
    assert auth.processar(None)[0].estado == Estado.NEGADO
    assert banco.listar_acessos()[0]["motivo"] == "pin_expirado"


def test_ministro_sem_pin_cadastrado_e_negado(ambiente):
    auth, pipe, rel, zonas, banco = ambiente
    m = banco.criar_usuario("Ministro", "", 3, None, None, [])
    auth.definir_zona(zonas[3])
    alimentar(auth, pipe, quadro(m.id, dist=0.2), 8)
    assert banco.listar_acessos()[0]["motivo"] == "pin_nao_cadastrado"


def test_bloqueio_apos_tres_negativas(ambiente, cfg):
    auth, pipe, rel, zonas, banco = ambiente
    servidor = banco.criar_usuario("Ana", "", 1, None, None, [])
    auth.definir_zona(zonas[3])
    for _ in range(3):
        alimentar(auth, pipe, quadro(servidor.id), 8)
        liberar_terminal(auth, pipe, rel)
    pipe.proximo = quadro(servidor.id)
    sit, res = auth.processar(None)
    assert sit.estado == Estado.BLOQUEADO and res is None and sit.bloqueio_restante_s > 0
    assert any(r["evento"] == "BLOQUEIO" for r in banco.listar_acessos())
    rel.t += cfg.bloqueio_s + 1
    pipe.proximo = quadro(problema="sem_rosto")
    assert auth.processar(None)[0].estado == Estado.AGUARDANDO


def test_so_decide_de_novo_depois_que_o_rosto_sai(ambiente):
    auth, pipe, rel, zonas, banco = ambiente
    u = banco.criar_usuario("Ana", "", 1, None, None, [])
    auth.definir_zona(zonas[1])
    alimentar(auth, pipe, quadro(u.id), 3)
    rel.t += 1.2
    for _ in range(10):                       # o rosto continua na frente da câmera
        sit = alimentar(auth, pipe, quadro(u.id))
    assert len([r for r in banco.listar_acessos() if r["evento"] == "CONCEDIDO"]) == 1
    assert sit.mensagem.startswith("Afaste-se")


def test_varios_rostos_pausa_a_analise(ambiente):
    auth, pipe, rel, zonas, banco = ambiente
    u = banco.criar_usuario("Ana", "", 1, None, None, [])
    auth.definir_zona(zonas[1])
    alimentar(auth, pipe, quadro(u.id), 2)
    sit = alimentar(auth, pipe, quadro(u.id, problema="varios_rostos"))
    assert "uma pessoa" in sit.mensagem.lower() and sit.progresso == 0
    assert alimentar(auth, pipe, quadro(u.id), 2).estado == Estado.ANALISANDO


def test_cadeia_do_log_permanece_integra(ambiente):
    auth, pipe, rel, zonas, banco = ambiente
    u = banco.criar_usuario("Ana", "", 1, None, None, [])
    auth.definir_zona(zonas[1])
    alimentar(auth, pipe, quadro(u.id), 3)
    liberar_terminal(auth, pipe, rel)
    auth.definir_zona(zonas[3])
    alimentar(auth, pipe, quadro(u.id), 8)
    assert banco.verificar_integridade().ok
