"""Relatórios de acesso: consulta, resumo, exportação em CSV e relatório HTML."""
from __future__ import annotations

import base64
import csv
import html
import io
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

from .banco import Banco

COLUNAS = ["id", "ts", "zona", "nivel_zona", "usuario_nome", "evento", "motivo", "distancia"]


def resumir(registros: list[dict]) -> dict:
    """Totais por evento, zona, usuário, motivo de negação e hora do dia."""
    return {
        "total": len(registros),
        "por_evento": dict(Counter(r["evento"] for r in registros)),
        "por_zona": dict(Counter(r["zona"] or "(sem zona)" for r in registros
                                 if r["evento"] in ("CONCEDIDO", "NEGADO"))),
        "por_usuario": dict(Counter(r["usuario_nome"] for r in registros
                                    if r["evento"] == "CONCEDIDO" and r["usuario_nome"])),
        "negacoes_por_motivo": dict(Counter(r["motivo"] for r in registros if r["evento"] == "NEGADO")),
        "por_hora": dict(sorted(Counter(r["ts"][11:13] for r in registros
                                        if r["evento"] in ("CONCEDIDO", "NEGADO")).items())),
    }


def exportar_csv(registros: list[dict], caminho: str | Path) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", newline="", encoding="utf-8-sig") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS, delimiter=";", extrasaction="ignore")
        escritor.writeheader()
        escritor.writerows(registros)
    return caminho


def _grafico_base64(resumo: dict) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, eixos = plt.subplots(1, 2, figsize=(8, 3), dpi=110)
    eventos = resumo["por_evento"]
    cores = {"CONCEDIDO": "#2e8b57", "NEGADO": "#c0392b", "BLOQUEIO": "#e08a1e"}
    eixos[0].bar(list(eventos), list(eventos.values()), color=[cores.get(e, "#777") for e in eventos])
    eixos[0].set_title("Eventos")
    horas = resumo["por_hora"]
    eixos[1].bar(list(horas), list(horas.values()), color="#1f5fa8")
    eixos[1].set_title("Decisões por hora")
    for eixo in eixos:
        eixo.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _tabela(titulo: str, dados: dict) -> str:
    if not dados:
        return f"<h3>{html.escape(titulo)}</h3><p>Sem dados.</p>"
    linhas = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{v}</td></tr>" for k, v in
                     sorted(dados.items(), key=lambda kv: -kv[1]))
    return f"<h3>{html.escape(titulo)}</h3><table><tr><th>Item</th><th>Qtde</th></tr>{linhas}</table>"


def gerar_html(banco: Banco, caminho: str | Path, inicio: Optional[str] = None,
               fim: Optional[str] = None, limite_tabela: int = 100) -> Path:
    registros = banco.listar_acessos(limite=0, inicio=inicio, fim=fim)
    resumo = resumir(registros)
    integridade = banco.verificar_integridade()
    grafico = _grafico_base64(resumo) if registros else None
    periodo = f"{inicio or 'início'} a {fim or 'hoje'}"
    cor = "#2e8b57" if integridade.ok else "#c0392b"
    linhas = "".join(
        "<tr>" + "".join(f"<td>{html.escape('' if r[c] is None else (f'{r[c]:.3f}' if c == 'distancia' else str(r[c])))}</td>"
                         for c in COLUNAS) + "</tr>" for r in registros[:limite_tabela])
    pagina = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<title>Relatório de acessos - BioCofre</title>
<style>body{{font-family:Arial,sans-serif;margin:2rem;color:#222}}table{{border-collapse:collapse;margin-bottom:1rem}}
td,th{{border:1px solid #ccc;padding:4px 10px;font-size:.9rem}}th{{background:#eef2f7}}
.sel{{display:inline-block;padding:4px 10px;border-radius:4px;color:#fff;background:{cor}}}</style></head><body>
<h1>Relatório de acessos - BioCofre</h1>
<p>Período: {html.escape(periodo)} | Gerado em {datetime.now():%d/%m/%Y %H:%M} | Registros: {resumo['total']}</p>
<p><span class="sel">{html.escape(integridade.mensagem)}</span></p>
{f'<img alt="Gráficos" src="data:image/png;base64,{grafico}">' if grafico else ''}
{_tabela('Eventos', resumo['por_evento'])}{_tabela('Decisões por zona', resumo['por_zona'])}
{_tabela('Acessos concedidos por usuário', resumo['por_usuario'])}
{_tabela('Negações por motivo', resumo['negacoes_por_motivo'])}
<h3>Últimos {min(limite_tabela, len(registros))} registros</h3>
<table><tr>{''.join(f'<th>{c}</th>' for c in COLUNAS)}</tr>{linhas}</table></body></html>"""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(pagina, encoding="utf-8")
    return caminho
