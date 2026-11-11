from biocofre.cli import principal


def test_usuarios_vazio_e_integridade(tmp_path, capsys):
    assert principal(["--dados", str(tmp_path), "usuarios"]) == 0
    assert "Nenhum usuário" in capsys.readouterr().out
    assert principal(["--dados", str(tmp_path), "integridade"]) == 0
    assert "0 registros" in capsys.readouterr().out


def test_relatorio_gera_arquivos(tmp_path, capsys):
    saida = tmp_path / "rel"
    assert principal(["--dados", str(tmp_path), "relatorio", "--saida", str(saida)]) == 0
    assert len(list(saida.glob("*.csv"))) == 1 and len(list(saida.glob("*.html"))) == 1
