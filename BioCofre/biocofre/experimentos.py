"""Reproduz os experimentos descritos na dissertação (comando ``experimentos``).

Divisão por pessoas: as primeiras servem para ESCOLHER parâmetros e CALIBRAR limiares
(conjunto de desenvolvimento); as demais, que o sistema nunca viu, servem para MEDIR o
resultado final (conjunto de teste). Assim os números finais não são otimistas.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import avaliacao as av
from .config import Config
from .pipeline.aquisicao import carregar_imagem
from .pipeline.classificacao import Galeria
from .pipeline.executor import Pipeline

VARIANTES = {
    "Configuração final": {},
    "Com CLAHE no descritor": dict(equalizar_caracteristicas=True),
    "Sem suavização gaussiana": dict(gauss_kernel=0),
    "Sem máscara elíptica": dict(mascara_eliptica=False),
    "Margem de recorte 0,10": dict(margem_face=0.10),
    "Margem de recorte 0,30": dict(margem_face=0.30),
    "Grade 4x4": dict(lbp_grade=4),
    "Grade 6x6": dict(lbp_grade=6),
    "Grade 10x10": dict(lbp_grade=10),
}
METRICAS = {"Qui-quadrado (adotada)": "chi2", "Euclidiana": "euclidiana", "Interseção de histogramas": "intersecao"}


def _arredondar_para_baixo(x: float, casas: int = 3) -> float:
    f = 10 ** casas
    return float(np.floor(x * f) / f)


def medir_tempos(base: av.Base, cfg: Config, n_quadros: int = 60) -> dict:
    """Tempo médio de cada fase em quadros de 640x480 (galeria com 5 modelos por pessoa)."""
    pipeline = Pipeline(cfg)
    galeria = Galeria(cfg.lbp_grade, 59, pipeline.pesos)
    for i, caminho in enumerate(base.caminhos):
        if i % 2 == 0 and len(galeria) < 100:
            r = pipeline.extrair(av.simular_quadro(carregar_imagem(caminho)))
            if r.vetor is not None:
                galeria.adicionar(int(base.pessoas[i]), r.vetor)
    soma: dict[str, list[float]] = {}
    passo = max(1, len(base.caminhos) // n_quadros)
    for caminho in base.caminhos[::passo][:n_quadros]:
        quadro = av.quadro_640x480(carregar_imagem(caminho))
        t0 = time.perf_counter()
        res = pipeline.processar(quadro, galeria)
        total = (time.perf_counter() - t0) * 1000.0
        for fase, ms in res.tempos_ms.items():
            soma.setdefault(fase, []).append(ms)
        soma.setdefault("total", []).append(total)
    medias = {k: float(np.mean(v)) for k, v in soma.items()}
    medias["modelos_na_galeria"] = len(galeria)
    medias["quadros_por_segundo"] = 1000.0 / medias["total"]
    return medias


def figuras_extras(resumo: dict, saida: Path) -> list[Path]:
    """Gráficos de ablação, de número de imagens de cadastro e de FPIR por tamanho do cadastro."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saida.mkdir(parents=True, exist_ok=True)
    azul, laranja, cinza = "#1f5fa8", "#d9692a", "#8a8a8a"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    arquivos = []

    nomes = list(resumo["ablacao"])
    valores = [resumo["ablacao"][n]["rank1"] * 100 for n in nomes]
    erros = [resumo["ablacao"][n]["rank1_dp"] * 100 for n in nomes]
    fig, ax = plt.subplots(figsize=(6.8, 3.9), dpi=200)
    cores = [azul if n == "Configuração final" else cinza for n in nomes]
    pos = np.arange(len(nomes))[::-1]
    ax.barh(pos, valores, xerr=erros, color=cores, error_kw=dict(ecolor="#333333", lw=0.8, capsize=2))
    ax.set_yticks(pos)
    ax.set_yticklabels(nomes, fontsize=8.5)
    ax.set_xlim(90, 100)
    ax.set_xlabel("Identificação correta no primeiro posto (%) - média e desvio dos sorteios")
    for p, v in zip(pos, valores):
        ax.text(v + 0.9, p, f"{v:.1f}", va="center", fontsize=8)
    fig.tight_layout()
    arquivos.append(saida / "fig_ablacao.png")
    fig.savefig(arquivos[-1])
    plt.close(fig)

    por_n = resumo["teste"]["rank1_por_n_treino"]
    xs = [int(k) for k in por_n]
    ys = [por_n[k]["rank1"] * 100 for k in por_n]
    dp = [por_n[k]["rank1_dp"] * 100 for k in por_n]
    fig, ax = plt.subplots(figsize=(5.6, 3.5), dpi=200)
    ax.errorbar(xs, ys, yerr=dp, marker="o", color=azul, capsize=3)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8)
    ax.set_xlabel("Imagens de cadastro por pessoa")
    ax.set_ylabel("Identificação correta (%)")
    ax.set_ylim(60, 101)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    arquivos.append(saida / "fig_amostras_cadastro.png")
    fig.savefig(arquivos[-1])
    plt.close(fig)

    pg = resumo["teste"]["fpir_por_tamanho_do_cadastro"]
    xs = [int(k) for k in pg]
    fig, ax = plt.subplots(figsize=(5.6, 3.5), dpi=200)
    ax.plot(xs, [pg[k]["verificacao_n2"] * 100 for k in pg], marker="o", color=laranja,
            label="Limiar calibrado por FAR (1:1)")
    ax.plot(xs, [pg[k]["identificacao_n2"] * 100 for k in pg], marker="s", color=azul,
            label="Limiar calibrado por FPIR (1:N)")
    ax.set_xlabel("Pessoas cadastradas")
    ax.set_ylabel("Intrusos aceitos (%) - limiar do Nível 2")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    arquivos.append(saida / "fig_fpir_cadastro.png")
    fig.savefig(arquivos[-1])
    plt.close(fig)
    return arquivos


def executar(raiz_base: str | Path, saida: str | Path, dev: str = "1-20", teste: str = "21-40",
             reps: int = 30, seed: int = 0, log=print, reps_calibracao: int = 200,
             reps_teste: int = 100) -> dict:
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    base_dev = av.carregar_base(raiz_base, av.faixa(dev))
    base_teste = av.carregar_base(raiz_base, av.faixa(teste))
    resumo: dict = {"dev": {"pessoas": len(base_dev.nomes), "imagens": len(base_dev.caminhos)},
                    "teste": {"pessoas": len(base_teste.nomes), "imagens": len(base_teste.caminhos)},
                    "reps": reps, "reps_calibracao": reps_calibracao, "reps_teste": reps_teste}
    cfg = Config()

    # 1) Ablação no conjunto de desenvolvimento -------------------------------------------
    log("1/5 Ablação no conjunto de desenvolvimento...")
    ablacao = {}
    for nome, mudancas in VARIANTES.items():
        r = av.avaliar_variante(base_dev, Config(**mudancas), "quadro", reps=reps, seed=seed)
        ablacao[nome] = {k: r[k] for k in ("deteccao", "rank1", "rank1_dp", "eer", "tempo_ms")}
        log(f"   {nome:28s} rank-1 {r['rank1']*100:5.1f}%  EER {r['eer']*100:4.1f}%")
    metricas = {}
    for nome, m in METRICAS.items():
        r = av.avaliar_variante(base_dev, cfg, "quadro", metrica=m, reps=reps, seed=seed)
        metricas[nome] = {k: r[k] for k in ("rank1", "rank1_dp", "eer")}
    resumo["ablacao"], resumo["metricas"] = ablacao, metricas
    (saida / "resumo_parcial.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")

    # 2) Calibração dos limiares no conjunto de desenvolvimento ------------------------------
    log("2/5 Calibração dos limiares (1:1 e 1:N)...")
    dev = av.avaliar_variante(base_dev, cfg, "quadro", reps=reps, seed=seed)
    n_int = 5
    # A) calibrado pela taxa de falsa aceitação de verificação (1:1)
    lim_ver = {n: _arredondar_para_baixo(t) for n, t in av.calibrar_limiares(dev["imp"]).items()}
    # B) calibrado pelo melhor candidato de intrusos contra um cadastro de 15 pessoas (1:N)
    d_int = av.distancias_intrusos(dev["D"], base_dev.pessoas, dev["detectado"], cfg.razao_max,
                                   n_intrusos=n_int, reps=reps_calibracao, seed=seed)
    lim_id = {n: _arredondar_para_baixo(t) for n, t in av.calibrar_limiares(d_int).items()}
    razoes = {}
    for razao in (1.0, 0.95, 0.9, 0.85, 0.8):
        razoes[str(razao)] = av.conjunto_aberto(dev["D"], base_dev.pessoas, dev["detectado"], lim_ver[2],
                                                razao, n_intrusos=n_int, reps=reps, seed=seed)
    resumo["calibracao"] = {"limiares_verificacao": lim_ver, "limiares_identificacao": lim_id,
                            "far_alvo": av.FAR_ALVO_PADRAO, "razao_max": cfg.razao_max,
                            "comparacoes_genuinas": int(dev["gen"].size),
                            "comparacoes_impostoras": int(dev["imp"].size),
                            "tentativas_intruso": int(d_int.size), "eer_dev": dev["eer"]}
    resumo["razao_dev_nivel2"] = razoes

    # 3) Teste em pessoas que o sistema nunca viu ------------------------------------------
    log("3/5 Teste em pessoas novas...")
    t = av.avaliar_variante(base_teste, cfg, "quadro", reps=reps_teste, seed=seed)
    far, frr = av.taxas_erro(t["gen"], t["imp"], np.array([lim_ver[n] for n in (1, 2, 3)]))
    abertos_ver, abertos_id = {}, {}
    for nivel in (1, 2, 3):
        abertos_ver[str(nivel)] = av.conjunto_aberto(t["D"], base_teste.pessoas, t["detectado"], lim_ver[nivel],
                                                     1.0, n_intrusos=n_int, reps=reps_teste, seed=seed)
        abertos_id[str(nivel)] = av.conjunto_aberto(t["D"], base_teste.pessoas, t["detectado"], lim_id[nivel],
                                                    cfg.razao_max, n_intrusos=n_int, reps=reps_teste, seed=seed)
    por_galeria = {}
    for n_cad in (2, 5, 10, 15):
        por_galeria[str(n_cad)] = {
            "verificacao_n2": av.conjunto_aberto(t["D"], base_teste.pessoas, t["detectado"], lim_ver[2], 1.0,
                                                 n_intrusos=n_int, reps=reps_teste, seed=seed, n_cadastrados=n_cad)["intruso_aceito"],
            "identificacao_n2": av.conjunto_aberto(t["D"], base_teste.pessoas, t["detectado"], lim_id[2],
                                                   cfg.razao_max, n_intrusos=n_int, reps=reps_teste, seed=seed,
                                                   n_cadastrados=n_cad)["intruso_aceito"]}
    rec = av.avaliar_variante(base_teste, cfg, "recortado", reps=reps_teste, seed=seed)
    resumo["teste"].update({
        "deteccao": t["deteccao"], "rank1": t["rank1"], "rank1_dp": t["rank1_dp"], "eer": t["eer"],
        "limiar_eer": t["limiar_eer"],
        "far_nos_limiares": {str(n): float(f) for n, f in zip((1, 2, 3), far)},
        "frr_nos_limiares": {str(n): float(f) for n, f in zip((1, 2, 3), frr)},
        "comparacoes_impostoras": int(t["imp"].size), "comparacoes_genuinas": int(t["gen"].size),
        "aberto_limiares_verificacao": abertos_ver, "aberto_limiares_identificacao": abertos_id,
        "fpir_por_tamanho_do_cadastro": por_galeria,
        "somente_classificador": {"rank1": rec["rank1"], "rank1_dp": rec["rank1_dp"], "eer": rec["eer"]},
    })
    limiares = lim_id

    # 4) Efeito do número de imagens de cadastro ------------------------------------------
    log("4/5 Número de imagens de cadastro...")
    por_n = {}
    for n in (1, 2, 3, 5, 7):
        m, dp = av.identificacao_rank1(t["D"], base_teste.pessoas, t["detectado"], n, reps_teste, seed)
        por_n[str(n)] = {"rank1": m, "rank1_dp": dp}
    resumo["teste"]["rank1_por_n_treino"] = por_n

    # 5) Tempos e gráficos --------------------------------------------------------------
    log("5/5 Tempos por fase e gráficos...")
    resumo["tempos"] = medir_tempos(base_teste, cfg)
    figuras = av.gerar_graficos(t["gen"], t["imp"], lim_ver, saida / "figuras")
    figuras += figuras_extras(resumo, saida / "figuras")
    resumo["figuras"] = [str(f) for f in figuras]
    resumo["config"] = asdict(cfg)
    (saida / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Resumo gravado em {saida / 'resumo.json'}")
    return resumo
