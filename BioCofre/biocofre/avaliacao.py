"""Avaliação biométrica do BioCofre: identificação, verificação, conjunto aberto e calibração.

Protocolo: as imagens de cada pessoa são divididas em cadastro (``n_treino`` imagens) e teste
(as demais), e o sorteio é repetido várias vezes. Uma imagem de cadastro nunca é usada como
teste. Todas as distâncias entre pares de imagens são calculadas uma única vez (matriz D) e
reaproveitadas em cada sorteio.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from .config import Config
from .pipeline.aquisicao import carregar_imagem, listar_imagens
from .pipeline.classificacao import distancias
from .pipeline.executor import Pipeline

FAR_ALVO_PADRAO = {1: 1e-2, 2: 1e-3, 3: 0.0}


# ---------------------------------------------------------------------------
# Base de imagens
# ---------------------------------------------------------------------------
@dataclass
class Base:
    caminhos: list[Path]
    pessoas: np.ndarray      # índice da pessoa de cada imagem
    nomes: list[str]


def _chave_natural(p: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


def carregar_base(raiz: str | Path, pessoas: Optional[Iterable[int]] = None) -> Base:
    """Cada subpasta de ``raiz`` é uma pessoa. ``pessoas`` filtra por posição (começando em 1)."""
    pastas = sorted((p for p in Path(raiz).iterdir() if p.is_dir()), key=_chave_natural)
    if pessoas is not None:
        escolhidas = set(pessoas)
        pastas = [p for i, p in enumerate(pastas, 1) if i in escolhidas]
    caminhos: list[Path] = []
    ids: list[int] = []
    nomes: list[str] = []
    for pasta in pastas:
        imagens = listar_imagens(pasta)
        if imagens:
            nomes.append(pasta.name)
            caminhos += imagens
            ids += [len(nomes) - 1] * len(imagens)
    if not caminhos:
        raise ValueError(f"Nenhuma imagem encontrada em '{raiz}'.")
    return Base(caminhos, np.array(ids), nomes)


def simular_quadro(imagem: np.ndarray, fator: int = 2, borda: int = 40) -> np.ndarray:
    """Transforma um retrato recortado em algo parecido com um quadro de câmera.

    Bases como a ORL já vêm recortadas no rosto e o detector precisa de contexto ao redor;
    por isso a imagem é ampliada e recebe uma borda que repete os pixels das extremidades.
    """
    grande = cv2.resize(imagem, None, fx=fator, fy=fator, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(grande, borda, borda, borda, borda, cv2.BORDER_REPLICATE)


def quadro_640x480(imagem: np.ndarray) -> np.ndarray:
    """Quadro de 640x480 com o retrato centralizado (usado para medir tempos realistas)."""
    altura = 300
    largura = int(imagem.shape[1] * altura / imagem.shape[0])
    rosto = cv2.resize(imagem, (largura, altura), interpolation=cv2.INTER_CUBIC)
    fundo = cv2.resize(imagem, (640, 480), interpolation=cv2.INTER_LINEAR)
    fundo = cv2.GaussianBlur(fundo, (0, 0), 25)
    x, y = (640 - largura) // 2, (480 - altura) // 2
    fundo[y:y + altura, x:x + largura] = rosto
    return fundo


def extrair_vetores(base: Base, cfg: Config, modo: str = "quadro"):
    """Vetores de características de todas as imagens.

    ``modo``: ``recortado`` (imagem já é o rosto) ou ``quadro`` (simula quadro e usa o detector).
    Devolve (vetores, detectado, pesos, tempo médio por imagem em ms).
    """
    pipeline = Pipeline(cfg)
    dim = cfg.lbp_grade ** 2 * 59
    vetores = np.zeros((len(base.caminhos), dim), np.float32)
    detectado = np.zeros(len(base.caminhos), bool)
    tempos = []
    for i, caminho in enumerate(base.caminhos):
        imagem = carregar_imagem(caminho)
        if modo == "quadro":
            imagem = simular_quadro(imagem)
        inicio = time.perf_counter()
        res = pipeline.extrair(imagem, ja_recortado=(modo == "recortado"))
        tempos.append((time.perf_counter() - inicio) * 1000.0)
        if res.vetor is not None:
            vetores[i] = res.vetor
            detectado[i] = True
    return vetores, detectado, pipeline.pesos, float(np.mean(tempos))


def matriz_distancias(vetores: np.ndarray, grade: int, pesos: np.ndarray,
                      metrica: str = "chi2") -> np.ndarray:
    n = len(vetores)
    v3 = vetores.reshape(n, grade * grade, -1)
    return np.stack([distancias(v3, v3[i], pesos, metrica) for i in range(n)]).astype(np.float32)


# ---------------------------------------------------------------------------
# Protocolo de sorteio
# ---------------------------------------------------------------------------
def sortear_cadastro(pessoas: np.ndarray, detectado: np.ndarray, n_treino: int,
                     rng: np.random.Generator, permitidas: Optional[set] = None) -> np.ndarray:
    gal = np.zeros(len(pessoas), bool)
    for p in np.unique(pessoas):
        if permitidas is not None and int(p) not in permitidas:
            continue
        idx = np.flatnonzero((pessoas == p) & detectado)
        if idx.size > n_treino:
            gal[rng.choice(idx, n_treino, replace=False)] = True
    return gal


def distancia_por_pessoa(D: np.ndarray, sondas: np.ndarray, gal: np.ndarray, pessoas: np.ndarray):
    """Matriz (sondas x pessoas cadastradas) com a menor distância aos modelos de cada pessoa."""
    cadastradas = np.unique(pessoas[gal])
    M = np.empty((len(sondas), len(cadastradas)), np.float32)
    for j, p in enumerate(cadastradas):
        colunas = np.flatnonzero(gal & (pessoas == p))
        M[:, j] = D[np.ix_(sondas, colunas)].min(axis=1)
    return M, cadastradas


def identificacao_rank1(D, pessoas, detectado, n_treino=5, reps=30, seed=0) -> tuple[float, float]:
    """Taxa de identificação correta no primeiro posto (conjunto fechado)."""
    rng = np.random.default_rng(seed)
    taxas = []
    for _ in range(reps):
        gal = sortear_cadastro(pessoas, detectado, n_treino, rng)
        sondas = np.flatnonzero(~gal & detectado & np.isin(pessoas, pessoas[gal]))
        M, cad = distancia_por_pessoa(D, sondas, gal, pessoas)
        taxas.append(float((cad[M.argmin(axis=1)] == pessoas[sondas]).mean()))
    return float(np.mean(taxas)), float(np.std(taxas))


def pontuacoes(D, pessoas, detectado, n_treino=5, reps=30, seed=0) -> tuple[np.ndarray, np.ndarray]:
    """Distâncias de tentativas legítimas (genuínas) e de impostores contra a pessoa alegada."""
    rng = np.random.default_rng(seed)
    gen, imp = [], []
    for _ in range(reps):
        gal = sortear_cadastro(pessoas, detectado, n_treino, rng)
        sondas = np.flatnonzero(~gal & detectado & np.isin(pessoas, pessoas[gal]))
        M, cad = distancia_por_pessoa(D, sondas, gal, pessoas)
        verdadeira = cad[None, :] == pessoas[sondas][:, None]
        gen.append(M[verdadeira])
        imp.append(M[~verdadeira])
    return np.concatenate(gen), np.concatenate(imp)


# ---------------------------------------------------------------------------
# Taxas de erro
# ---------------------------------------------------------------------------
def taxas_erro(gen: np.ndarray, imp: np.ndarray, limiares: np.ndarray):
    """FAR e FRR para cada limiar: aceita-se quando a distância é menor ou igual ao limiar."""
    g, i = np.sort(gen), np.sort(imp)
    far = np.searchsorted(i, limiares, side="right") / i.size
    frr = 1.0 - np.searchsorted(g, limiares, side="right") / g.size
    return far, frr


def eer(gen: np.ndarray, imp: np.ndarray) -> tuple[float, float]:
    """Taxa de erro igual e o limiar em que ela ocorre."""
    candidatos = np.unique(np.concatenate([gen, imp]))
    if candidatos.size > 4000:
        candidatos = np.quantile(candidatos, np.linspace(0, 1, 4000))
    far, frr = taxas_erro(gen, imp, candidatos)
    k = int(np.argmin(np.abs(far - frr)))
    return float((far[k] + frr[k]) / 2), float(candidatos[k])


def limiar_para_far(imp: np.ndarray, far_alvo: float) -> float:
    """Maior limiar cujo FAR observado não passa de ``far_alvo`` (0 = nenhum impostor aceito)."""
    s = np.sort(imp)
    finitos = s[np.isfinite(s)]
    k = int(np.floor(far_alvo * s.size))
    if finitos.size == 0:
        return 0.0
    if k >= finitos.size:
        return float(finitos.max())
    return float(np.nextafter(finitos[k], -np.inf))


def calibrar_limiares(distancias_impostoras: np.ndarray, far_alvo: Optional[dict] = None,
                      passo: float = 0.01) -> dict[int, float]:
    """Limiar de cada nível para a taxa de falsa aceitação desejada.

    Com poucos dados as metas de níveis vizinhos podem cair no mesmo valor; por isso cada nível
    é forçado a ser pelo menos ``passo`` mais rígido que o anterior.
    """
    alvos = far_alvo or FAR_ALVO_PADRAO
    saida: dict[int, float] = {}
    anterior = None
    for nivel in sorted(alvos):
        limiar = limiar_para_far(distancias_impostoras, alvos[nivel])
        if anterior is not None:
            limiar = min(limiar, anterior - passo)
        saida[nivel] = anterior = limiar
    return saida


# ---------------------------------------------------------------------------
# Conjunto aberto: intrusos que não estão no cadastro
# ---------------------------------------------------------------------------
def conjunto_aberto(D, pessoas, detectado, limiar: float, razao_max: float = 1.0,
                    n_treino: int = 5, n_intrusos: int = 5, reps: int = 30, seed: int = 0,
                    n_cadastrados: Optional[int] = None) -> dict:
    """Aplica a regra de decisão (limiar e razão) a legítimos e a intrusos desconhecidos.

    Cada sorteio separa ``n_intrusos`` pessoas que NÃO entram no cadastro e cadastra as demais
    (ou apenas ``n_cadastrados`` delas). Um intruso é aceito se o melhor candidato do cadastro
    passar pelo limiar e pela razão d1/d2: é a taxa de falsa identificação positiva (FPIR).
    """
    rng = np.random.default_rng(seed)
    ids = np.unique(pessoas)
    linhas = []
    for _ in range(reps):
        perm = rng.permutation(ids)
        intrusos, cadastrados = perm[:n_intrusos], perm[n_intrusos:]
        if n_cadastrados is not None:
            cadastrados = cadastrados[:n_cadastrados]
        gal = sortear_cadastro(pessoas, detectado, n_treino, rng, set(int(c) for c in cadastrados))
        s_leg = np.flatnonzero(~gal & detectado & np.isin(pessoas, np.unique(pessoas[gal])))
        s_int = np.flatnonzero(detectado & np.isin(pessoas, intrusos))
        sondas = np.concatenate([s_leg, s_int])
        eh_intruso = np.concatenate([np.zeros(len(s_leg), bool), np.ones(len(s_int), bool)])
        M, cad = distancia_por_pessoa(D, sondas, gal, pessoas)
        ordem = np.argsort(M, axis=1)
        linha = np.arange(len(M))
        d1 = M[linha, ordem[:, 0]]
        d2 = M[linha, ordem[:, 1]] if M.shape[1] > 1 else np.full(len(M), np.inf)
        id1 = cad[ordem[:, 0]]
        aceito = (d1 <= limiar) & ((d1 / np.maximum(d2, 1e-12)) <= razao_max)
        leg = ~eh_intruso
        correto = aceito & leg & (id1 == pessoas[sondas])
        troca = aceito & leg & (id1 != pessoas[sondas])
        rejeitado = ~aceito & leg
        intruso_aceito = aceito & eh_intruso
        linhas.append([correto.sum() / leg.sum(), troca.sum() / leg.sum(),
                       rejeitado.sum() / leg.sum(), intruso_aceito.sum() / eh_intruso.sum(),
                       eh_intruso.sum(), leg.sum()])
    a = np.array(linhas)
    return {
        "acesso_correto": float(a[:, 0].mean()),
        "troca_de_identidade": float(a[:, 1].mean()),
        "legitimo_rejeitado": float(a[:, 2].mean()),
        "intruso_aceito": float(a[:, 3].mean()),
        "intruso_aceito_dp": float(a[:, 3].std()),
        "tentativas_intruso": int(a[:, 4].sum()),
        "tentativas_legitimas": int(a[:, 5].sum()),
    }


def distancias_intrusos(D, pessoas, detectado, razao_max: float = 1.0, n_treino: int = 5,
                        n_intrusos: int = 5, reps: int = 30, seed: int = 0,
                        n_cadastrados: Optional[int] = None) -> np.ndarray:
    """Melhor distância de cada tentativa de intruso contra um cadastro sorteado.

    Tentativas barradas pela razão d1/d2 valem infinito. O limiar que mantém a fração desejada
    destas distâncias abaixo dele é o limiar para um dado FPIR (calibração 1:N).
    """
    rng = np.random.default_rng(seed)
    ids = np.unique(pessoas)
    saida = []
    for _ in range(reps):
        perm = rng.permutation(ids)
        intrusos, cadastrados = perm[:n_intrusos], perm[n_intrusos:]
        if n_cadastrados is not None:
            cadastrados = cadastrados[:n_cadastrados]
        gal = sortear_cadastro(pessoas, detectado, n_treino, rng, set(int(c) for c in cadastrados))
        s_int = np.flatnonzero(detectado & np.isin(pessoas, intrusos))
        M, _ = distancia_por_pessoa(D, s_int, gal, pessoas)
        ordem = np.sort(M, axis=1)
        d1 = ordem[:, 0]
        d2 = ordem[:, 1] if M.shape[1] > 1 else np.full(len(M), np.inf)
        barrado = (d1 / np.maximum(d2, 1e-12)) > razao_max
        saida.append(np.where(barrado, np.inf, d1))
    return np.concatenate(saida)


# ---------------------------------------------------------------------------
# Experimentos e gráficos
# ---------------------------------------------------------------------------
def avaliar_variante(base: Base, cfg: Config, modo: str, metrica: str = "chi2", n_treino: int = 5,
                     reps: int = 30, seed: int = 0) -> dict:
    vetores, detectado, pesos, ms = extrair_vetores(base, cfg, modo)
    D = matriz_distancias(vetores, cfg.lbp_grade, pesos, metrica)
    r1, dp = identificacao_rank1(D, base.pessoas, detectado, n_treino, reps, seed)
    gen, imp = pontuacoes(D, base.pessoas, detectado, n_treino, reps, seed)
    e, tau = eer(gen, imp)
    return {"deteccao": float(detectado.mean()), "rank1": r1, "rank1_dp": dp, "eer": e,
            "limiar_eer": tau, "tempo_ms": ms, "D": D, "detectado": detectado,
            "gen": gen, "imp": imp}


def gerar_graficos(gen, imp, limiares: dict, saida: str | Path) -> list[Path]:
    """Histograma das distâncias, curvas FAR/FRR e curva ROC."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    azul, laranja, cinza = "#1f5fa8", "#d9692a", "#555555"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    arquivos = []

    fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=200)
    bins = np.linspace(min(gen.min(), imp.min()), max(gen.max(), imp.max()), 70)
    ax.hist(imp, bins=bins, color=laranja, alpha=0.75, density=True, label="Impostores")
    ax.hist(gen, bins=bins, color=azul, alpha=0.75, density=True, label="Legítimos")
    for nivel, tau in sorted(limiares.items()):
        ax.axvline(tau, color=cinza, ls="--", lw=1)
        ax.text(tau, ax.get_ylim()[1] * 0.97, f"N{nivel}", ha="right", va="top", fontsize=8, color=cinza)
    ax.set_xlabel("Distância qui-quadrado média por região")
    ax.set_ylabel("Densidade")
    ax.legend(frameon=False)
    fig.tight_layout()
    arquivos.append(saida / "fig_distribuicao_distancias.png")
    fig.savefig(arquivos[-1])
    plt.close(fig)

    candidatos = np.linspace(min(gen.min(), imp.min()), max(gen.max(), imp.max()), 600)
    far, frr = taxas_erro(gen, imp, candidatos)
    e, tau = eer(gen, imp)
    fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=200)
    ax.plot(candidatos, far * 100, color=laranja, label="FAR (falsa aceitação)")
    ax.plot(candidatos, frr * 100, color=azul, label="FRR (falsa rejeição)")
    ax.plot([tau], [e * 100], "o", color="black")
    ax.annotate(f"EER = {e * 100:.1f}%", (tau, e * 100), textcoords="offset points",
                xytext=(10, 10), fontsize=9)
    ax.set_xlabel("Limiar de distância")
    ax.set_ylabel("Taxa de erro (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    arquivos.append(saida / "fig_far_frr.png")
    fig.savefig(arquivos[-1])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.8), dpi=200)
    ax.plot(np.maximum(far, 1e-5) * 100, (1 - frr) * 100, color=azul)
    ax.set_xscale("log")
    ax.set_xlabel("FAR (%) - escala logarítmica")
    ax.set_ylabel("Aceitação de legítimos (1 - FRR, %)")
    ax.set_ylim(0, 101)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    arquivos.append(saida / "fig_roc.png")
    fig.savefig(arquivos[-1])
    plt.close(fig)
    return arquivos


def faixa(texto: str) -> list[int]:
    """Converte '21-40' ou '1,3,5' em lista de posições."""
    posicoes: list[int] = []
    for parte in texto.split(","):
        if "-" in parte:
            a, b = parte.split("-")
            posicoes += list(range(int(a), int(b) + 1))
        elif parte.strip():
            posicoes.append(int(parte))
    return posicoes
