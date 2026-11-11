import numpy as np
import pytest

from biocofre.banco import Banco, ErroBanco


def _vetor():
    return np.random.default_rng(0).random(64 * 59).astype(np.float32)


def test_criar_usuario_e_modelos(banco):
    u = banco.criar_usuario("Ana", "Analista", 1, None, None, [_vetor(), _vetor()])
    assert u.id and u.nivel == 1 and not u.tem_pin
    modelos = banco.carregar_modelos()
    assert len(modelos) == 2 and modelos[0][0] == u.id and np.allclose(modelos[0][1], _vetor())


def test_nome_duplicado_e_nivel_invalido(banco):
    banco.criar_usuario("Ana", "", 1, None, None, [])
    with pytest.raises(ErroBanco):
        banco.criar_usuario("Ana", "", 2, None, None, [])
    with pytest.raises(Exception):
        banco.criar_usuario("Beto", "", 7, None, None, [])


def test_pin_guardado_com_hash_e_verificado(banco):
    u = banco.criar_usuario("Min", "", 3, None, "1234", [])
    assert u.tem_pin and banco.verificar_pin(u.id, "1234") and not banco.verificar_pin(u.id, "0000")
    bruto = banco._con.execute("SELECT pin_hash FROM usuarios").fetchone()["pin_hash"]
    assert b"1234" not in bruto and len(bruto) == 32


def test_remover_usuario_apaga_modelos(banco):
    u = banco.criar_usuario("Ana", "", 1, None, None, [_vetor()])
    banco.remover_usuario(u.id)
    assert banco.carregar_modelos() == [] and banco.obter_usuario(u.id) is None


def test_usuario_desativado_sai_da_galeria(banco):
    u = banco.criar_usuario("Ana", "", 1, None, None, [_vetor()])
    banco.definir_ativo(u.id, False)
    assert banco.carregar_modelos() == []


def test_zonas_padrao(banco):
    z = banco.zonas()
    assert [x.nivel for x in z] == [1, 2, 2, 3] and z[-1].nome == "Cofre de toxinas"


def test_log_encadeado_integro_e_deteccao_de_adulteracao(banco):
    for i in range(5):
        banco.registrar_acesso("CONCEDIDO", 1, None, f"U{i}", "ok", 0.3)
    assert banco.verificar_integridade().ok
    banco._con.execute("UPDATE acessos SET evento = 'NEGADO' WHERE id = 3")
    banco._con.commit()
    r = banco.verificar_integridade()
    assert not r.ok and r.primeiro_invalido == 3


def test_remocao_de_registro_do_meio_quebra_a_cadeia(banco):
    for i in range(5):
        banco.registrar_acesso("CONCEDIDO", 1, None, f"U{i}", "ok", 0.3)
    banco._con.execute("DELETE FROM acessos WHERE id = 3")
    banco._con.commit()
    r = banco.verificar_integridade()
    assert not r.ok and r.primeiro_invalido == 4


def test_chave_diferente_nao_valida_a_cadeia(cfg, banco, tmp_path):
    banco.registrar_acesso("CONCEDIDO", 1, None, "Ana", "ok", 0.3)
    banco.fechar()
    outra = tmp_path / "outra.key"
    b2 = Banco(cfg.caminho_banco, outra, iteracoes_kdf=1000)
    assert not b2.verificar_integridade().ok
    b2.fechar()


def test_senha_do_administrador(banco):
    assert not banco.tem_senha_admin()
    with pytest.raises(ErroBanco):
        banco.definir_senha_admin("123")
    banco.definir_senha_admin("segredo123")
    assert banco.tem_senha_admin() and banco.verificar_senha_admin("segredo123")
    assert not banco.verificar_senha_admin("outra")


def test_listar_acessos_filtra(banco):
    banco.registrar_acesso("CONCEDIDO", 1, None, "Ana", "ok", 0.3, ts="2026-10-01T10:00:00")
    banco.registrar_acesso("NEGADO", 4, None, None, "face_nao_reconhecida", 0.5, ts="2026-10-02T10:00:00")
    assert len(banco.listar_acessos(evento="NEGADO")) == 1
    assert len(banco.listar_acessos(inicio="2026-10-02")) == 1
    assert len(banco.listar_acessos(fim="2026-10-01")) == 1
