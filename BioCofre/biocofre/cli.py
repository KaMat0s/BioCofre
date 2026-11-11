"""Interface de linha de comando do BioCofre (``python main.py <comando>``)."""
from __future__ import annotations

import argparse
import getpass
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from . import avaliacao as av
from . import relatorios
from .acesso import NOMES_NIVEL, autorizado, montar_galeria
from .banco import Banco, ErroBanco
from .cadastro import Cadastro, ErroCadastro, MENSAGENS_AMOSTRA
from .config import Config
from .pipeline.aquisicao import ErroAquisicao, abrir_fonte, carregar_imagem
from .pipeline.classificacao import decidir
from .pipeline.executor import Pipeline


def _config(args) -> Config:
    caminho = args.config if args.config else ("config.json" if Path("config.json").exists() else None)
    cfg = Config.carregar(caminho)
    if args.dados:
        cfg.pasta_dados = args.dados
    return cfg


def _banco(cfg: Config) -> Banco:
    return Banco(cfg.caminho_banco, cfg.caminho_chave)


def _exigir_admin(banco: Banco) -> None:
    senha = os.environ.get("BIOCOFRE_ADMIN_SENHA")
    if not banco.tem_senha_admin():
        print("Primeiro uso: defina a senha do administrador.")
        nova = senha or getpass.getpass("Nova senha (mínimo 6 caracteres): ")
        if not senha and nova != getpass.getpass("Repita a senha: "):
            raise SystemExit("As senhas não conferem.")
        banco.definir_senha_admin(nova)
        return
    if not banco.verificar_senha_admin(senha or getpass.getpass("Senha do administrador: ")):
        raise SystemExit("Senha do administrador incorreta.")


# ---------------------------------------------------------------------------
def cmd_gui(args) -> int:
    from .gui import executar_gui
    executar_gui(_config(args), args.origem)
    return 0


def cmd_cadastrar(args) -> int:
    cfg = _config(args)
    banco = _banco(cfg)
    _exigir_admin(banco)
    pipeline = Pipeline(cfg)
    cadastro = Cadastro(banco, pipeline, cfg)
    alvo = args.amostras or cfg.amostras_cadastro
    try:
        fonte = abrir_fonte(args.origem if args.origem is not None else cfg.camera_indice, cfg)
    except ErroAquisicao as exc:
        print(f"Erro: {exc}")
        return 1
    amostras, motivos, tentativas = [], {}, 0
    print(f"Capturando {alvo} amostras de '{args.nome}'. Olhe para a câmera e varie levemente a pose.")
    try:
        while len(amostras) < alvo and tentativas < alvo * 12:
            quadro = fonte.ler()
            if quadro is None:
                break
            tentativas += 1
            amostra = cadastro.avaliar_quadro(quadro, ja_recortado=args.recortado,
                                              verificar_qualidade=not args.sem_qualidade)
            motivos[amostra.motivo] = motivos.get(amostra.motivo, 0) + 1
            if amostra.aceita:
                amostras.append(amostra)
                print(f"  amostra {len(amostras)}/{alvo}")
                time.sleep(args.intervalo)
    finally:
        fonte.fechar()
    print("Resumo da captura:", ", ".join(f"{MENSAGENS_AMOSTRA.get(k, k)}: {v}" for k, v in motivos.items()))
    try:
        usuario = cadastro.cadastrar(args.nome, args.cargo, args.nivel, args.divisao, args.pin, amostras,
                                     montar_galeria(banco, pipeline), args.permitir_duplicado)
    except (ErroCadastro, ErroBanco) as exc:
        print(f"Cadastro não realizado: {exc}")
        return 1
    print(f"Usuário cadastrado: {usuario.nome} - {NOMES_NIVEL[usuario.nivel]} (id {usuario.id})")
    return 0


def cmd_usuarios(args) -> int:
    banco = _banco(_config(args))
    usuarios = banco.listar_usuarios()
    if not usuarios:
        print("Nenhum usuário cadastrado.")
    for u in usuarios:
        print(f"{u.id:>3}  {u.nome:<28} nível {u.nivel}  divisão: {u.divisao or '-':<14} "
              f"PIN: {'sim' if u.tem_pin else 'não'}  {'ativo' if u.ativo else 'DESATIVADO'}  {u.cargo}")
    return 0


def cmd_remover(args) -> int:
    banco = _banco(_config(args))
    _exigir_admin(banco)
    usuario = banco.obter_usuario(args.id)
    if not usuario:
        print("Usuário não encontrado.")
        return 1
    banco.remover_usuario(args.id)
    banco.registrar_acesso("REMOCAO", None, None, usuario.nome, "removido_pelo_administrador")
    print(f"Usuário '{usuario.nome}' removido.")
    return 0


def cmd_acessos(args) -> int:
    banco = _banco(_config(args))
    for r in reversed(banco.listar_acessos(limite=args.limite)):
        dist = "" if r["distancia"] is None else f"{r['distancia']:.3f}"
        print(f"{r['id']:>5} {r['ts']} {r['evento']:<9} {(r['zona'] or '-'):<34} "
              f"{(r['usuario_nome'] or '-'):<24} {r['motivo'] or '':<36} {dist}")
    return 0


def cmd_integridade(args) -> int:
    resultado = _banco(_config(args)).verificar_integridade()
    print(resultado.mensagem)
    return 0 if resultado.ok else 2


def cmd_relatorio(args) -> int:
    banco = _banco(_config(args))
    pasta = Path(args.saida)
    carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
    registros = banco.listar_acessos(limite=0, inicio=args.inicio, fim=args.fim)
    csv_ = relatorios.exportar_csv(registros, pasta / f"acessos_{carimbo}.csv")
    html_ = relatorios.gerar_html(banco, pasta / f"relatorio_{carimbo}.html", args.inicio, args.fim)
    print(f"CSV:  {csv_}\nHTML: {html_}")
    return 0


def cmd_identificar(args) -> int:
    """Diagnóstico: roda uma imagem pelas cinco fases, sem registrar acesso."""
    cfg = _config(args)
    banco = _banco(cfg)
    pipeline = Pipeline(cfg)
    galeria = montar_galeria(banco, pipeline)
    try:
        imagem = carregar_imagem(args.imagem)
    except ErroAquisicao as exc:
        print(f"Erro: {exc}")
        return 1
    res = pipeline.processar(imagem, galeria, ja_recortado=args.recortado)
    print("Tempos (ms):", {k: round(v, 1) for k, v in res.tempos_ms.items()})
    if res.problema:
        print(f"Sem resultado: {res.problema}")
        return 1
    nomes = {u.id: u for u in banco.listar_usuarios()}
    zona = banco.obter_zona(args.zona) if args.zona else banco.zonas()[0]
    limiar = cfg.limiar_do_nivel(zona.nivel)
    print(f"Zona: {zona.nome} (nível {zona.nivel}) - limiar {limiar:.3f}, razão máx. {cfg.razao_max}")
    for c in res.ranking[:5]:
        print(f"  {nomes[c.usuario_id].nome:<28} distância {c.distancia:.4f}")
    dec = decidir(res.ranking, limiar, cfg.razao_max)
    if dec.aceito:
        usuario = nomes[dec.usuario_id]
        ok, motivo = autorizado(usuario, zona)
        print(f"Identificado: {usuario.nome} -> {'AUTORIZADO' if ok else 'NÃO AUTORIZADO (' + motivo + ')'}")
    else:
        print(f"Rosto não aceito ({dec.motivo}).")
    return 0


def cmd_avaliar(args) -> int:
    cfg = _config(args)
    base = av.carregar_base(args.base, av.faixa(args.pessoas) if args.pessoas else None)
    print(f"{len(base.nomes)} pessoas, {len(base.caminhos)} imagens, modo '{args.modo}'.")
    vetores, detectado, pesos, ms = av.extrair_vetores(base, cfg, args.modo)
    D = av.matriz_distancias(vetores, cfg.lbp_grade, pesos)
    r1, dp = av.identificacao_rank1(D, base.pessoas, detectado, args.treino, args.reps)
    gen, imp = av.pontuacoes(D, base.pessoas, detectado, args.treino, args.reps)
    e, tau = av.eer(gen, imp)
    print(f"Detecção do rosto: {detectado.mean() * 100:.1f}%  |  tempo médio por imagem: {ms:.1f} ms")
    print(f"Identificação no primeiro posto ({args.treino} imagens de cadastro): {r1 * 100:.1f}% (+-{dp * 100:.1f})")
    print(f"EER: {e * 100:.2f}% no limiar {tau:.3f}")
    limiares = av.calibrar_limiares(imp)
    print("Limiares sugeridos por FAR (1:1):", {n: round(t, 3) for n, t in limiares.items()})
    if args.saida:
        for arquivo in av.gerar_graficos(gen, imp, limiares, args.saida):
            print("Figura:", arquivo)
    return 0


def cmd_calibrar(args) -> int:
    """Calibra os limiares com imagens do próprio ambiente (uma pasta por pessoa)."""
    cfg = _config(args)
    base = av.carregar_base(args.base)
    n = len(base.nomes)
    if n < 3:
        print("São necessárias pelo menos 3 pessoas para calibrar os limiares.")
        return 1
    vetores, detectado, pesos, _ = av.extrair_vetores(base, cfg, args.modo)
    D = av.matriz_distancias(vetores, cfg.lbp_grade, pesos)
    intrusos = max(1, n // 4)
    d_int = av.distancias_intrusos(D, base.pessoas, detectado, cfg.razao_max, args.treino,
                                   intrusos, args.reps, n_cadastrados=n - intrusos)
    limiares = {k: round(float(np.floor(v * 1000) / 1000), 3) for k, v in av.calibrar_limiares(d_int).items()}
    print(f"{n} pessoas ({intrusos} como intrusos em cada sorteio), {d_int.size} tentativas de intruso.")
    print("Limiares calibrados (FPIR alvo 1%, 0,1% e 0):", limiares)
    if args.gravar:
        cfg.limiar = limiares
        cfg.salvar(args.gravar)
        print(f"Configuração gravada em {args.gravar}")
    return 0


def cmd_experimentos(args) -> int:
    from .experimentos import executar
    executar(args.base, args.saida, reps=args.reps)
    return 0


def cmd_demo(args) -> int:
    """Cria usuários de demonstração e um roteiro de quadros simulados a partir de uma base (ex.: ORL).

    O roteiro mostra, um de cada vez, os usuários cadastrados (com imagens que não foram usadas no
    cadastro) e três intrusos, separados por quadros vazios. Serve para testar sem câmera.
    """
    cfg = _config(args)
    banco = _banco(cfg)
    if not banco.tem_senha_admin():
        banco.definir_senha_admin(os.environ.get("BIOCOFRE_ADMIN_SENHA", "admin123"))
    base = av.carregar_base(args.base, range(1, args.usuarios + 4))
    pipeline = Pipeline(cfg)
    cadastro = Cadastro(banco, pipeline, cfg)
    papeis = [("Ministro (demo)", "Ministro do Meio Ambiente", 3, None, "1234"),
              ("Diretor Toxicologia (demo)", "Diretor de divisão", 2, "Toxicologia", None),
              ("Diretor Biossegurança (demo)", "Diretor de divisão", 2, "Biossegurança", None)]
    pasta = Path(args.saida)
    pasta.mkdir(parents=True, exist_ok=True)
    vazio = np.full((384, 344, 3), 40, np.uint8)
    cv2.imwrite(str(pasta / "vazio.png"), vazio)
    roteiro = ["# fps=15"]
    for idx in range(args.usuarios + 3):
        arquivos = [c for c, p in zip(base.caminhos, base.pessoas) if p == idx]
        if idx < args.usuarios:
            nome, cargo, nivel, divisao, pin = (papeis[idx] if idx < len(papeis)
                                                else (f"Servidor {idx + 1} (demo)", "Servidor", 1, None, None))
            amostras = [cadastro.avaliar_quadro(av.simular_quadro(carregar_imagem(a)), verificar_qualidade=False)
                        for a in arquivos[:5]]
            cadastro.cadastrar(nome, cargo, nivel, divisao, pin, amostras, permitir_duplicado=True)
            print(f"  cadastrado: {nome} (nível {nivel})")
        sondas = arquivos[5:] if idx < args.usuarios else arquivos[:5]
        nomes = []
        for k, a in enumerate(sondas):
            nome_arq = f"p{idx + 1:02d}_{k}.png"
            cv2.imwrite(str(pasta / nome_arq), av.simular_quadro(carregar_imagem(a)))
            nomes.append(nome_arq)
        roteiro += [nomes[i % len(nomes)] for i in range(45)] + ["vazio.png"] * 120
    (pasta / "roteiro.txt").write_text("\n".join(roteiro), encoding="utf-8")
    print(f"Roteiro de demonstração em {pasta}. Execute: python main.py gui --origem {pasta}")
    return 0


# ---------------------------------------------------------------------------
def montar_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="biocofre", description="BioCofre - acesso por reconhecimento facial")
    p.add_argument("--config", help="arquivo JSON de configuração")
    p.add_argument("--dados", help="pasta de dados (banco e chave do log)")
    p.add_argument("--versao", action="version", version=f"BioCofre {__version__}")
    sub = p.add_subparsers(dest="comando")

    s = sub.add_parser("gui", help="abre a interface gráfica (padrão)")
    s.add_argument("--origem", help="índice da câmera, pasta de imagens, imagem ou vídeo")
    s.set_defaults(func=cmd_gui)

    s = sub.add_parser("cadastrar", help="cadastra um usuário capturando amostras")
    s.add_argument("--nome", required=True)
    s.add_argument("--cargo", default="")
    s.add_argument("--nivel", type=int, choices=(1, 2, 3), required=True)
    s.add_argument("--divisao")
    s.add_argument("--pin")
    s.add_argument("--origem")
    s.add_argument("--amostras", type=int)
    s.add_argument("--intervalo", type=float, default=0.3, help="segundos entre amostras")
    s.add_argument("--recortado", action="store_true", help="as imagens já são só o rosto")
    s.add_argument("--sem-qualidade", action="store_true", help="não verifica nitidez, brilho e contraste")
    s.add_argument("--permitir-duplicado", action="store_true")
    s.set_defaults(func=cmd_cadastrar)

    sub.add_parser("usuarios", help="lista os usuários").set_defaults(func=cmd_usuarios)
    s = sub.add_parser("remover", help="remove um usuário")
    s.add_argument("--id", type=int, required=True)
    s.set_defaults(func=cmd_remover)
    s = sub.add_parser("acessos", help="mostra o log de acessos")
    s.add_argument("--limite", type=int, default=30)
    s.set_defaults(func=cmd_acessos)
    sub.add_parser("integridade", help="verifica a cadeia do log").set_defaults(func=cmd_integridade)
    s = sub.add_parser("relatorio", help="gera CSV e HTML do log")
    s.add_argument("--saida", default="relatorios")
    s.add_argument("--inicio")
    s.add_argument("--fim")
    s.set_defaults(func=cmd_relatorio)

    s = sub.add_parser("identificar", help="roda uma imagem pelas cinco fases (diagnóstico)")
    s.add_argument("--imagem", required=True)
    s.add_argument("--zona", type=int)
    s.add_argument("--recortado", action="store_true")
    s.set_defaults(func=cmd_identificar)

    s = sub.add_parser("avaliar", help="mede identificação, EER e limiares em uma base de imagens")
    s.add_argument("--base", required=True, help="pasta com uma subpasta por pessoa")
    s.add_argument("--modo", choices=("quadro", "recortado"), default="quadro")
    s.add_argument("--pessoas", help="faixa de pessoas, ex.: 1-20")
    s.add_argument("--treino", type=int, default=5)
    s.add_argument("--reps", type=int, default=30)
    s.add_argument("--saida", help="pasta para os gráficos")
    s.set_defaults(func=cmd_avaliar)

    s = sub.add_parser("calibrar", help="calibra os limiares com imagens do seu ambiente")
    s.add_argument("--base", required=True)
    s.add_argument("--modo", choices=("quadro", "recortado"), default="recortado")
    s.add_argument("--treino", type=int, default=5)
    s.add_argument("--reps", type=int, default=200)
    s.add_argument("--gravar", help="grava os limiares neste JSON de configuração")
    s.set_defaults(func=cmd_calibrar)

    s = sub.add_parser("experimentos", help="reproduz os experimentos da dissertação")
    s.add_argument("--base", required=True)
    s.add_argument("--saida", default="resultados")
    s.add_argument("--reps", type=int, default=30)
    s.set_defaults(func=cmd_experimentos)

    s = sub.add_parser("demo", help="cria usuários e quadros de demonstração a partir de uma base")
    s.add_argument("--base", required=True)
    s.add_argument("--usuarios", type=int, default=6)
    s.add_argument("--saida", default="dados/demo_quadros")
    s.set_defaults(func=cmd_demo)
    return p


def principal(argv=None) -> int:
    parser = montar_parser()
    args = parser.parse_args(argv)
    if not args.comando:  # sem subcomando: abre a interface gráfica
        args = parser.parse_args((argv if argv is not None else sys.argv[1:]) + ["gui"])
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(principal())
