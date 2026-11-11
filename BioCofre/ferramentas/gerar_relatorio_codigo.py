"""Gera o Relatório de Código (PDF) com todas as linhas do programa numeradas.

Uso: python ferramentas/gerar_relatorio_codigo.py [saida.pdf]   (requer: pip install reportlab matplotlib)
"""
import ast, os, sys
from pathlib import Path
import matplotlib
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Preformatted,
                                PageBreak, Table, TableStyle, KeepTogether)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / 'Relatorio_de_Codigo.pdf'
fontes = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf'
pdfmetrics.registerFont(TTFont('Mono', str(fontes / 'DejaVuSansMono.ttf')))
pdfmetrics.registerFont(TTFont('Sans', str(fontes / 'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('Sans-Bold', str(fontes / 'DejaVuSans-Bold.ttf')))

ARQUIVOS = ['main.py', 'requirements.txt', 'biocofre/__init__.py', 'biocofre/config.py',
            'biocofre/pipeline/aquisicao.py', 'biocofre/pipeline/preprocessamento.py',
            'biocofre/pipeline/segmentacao.py', 'biocofre/pipeline/caracteristicas.py',
            'biocofre/pipeline/classificacao.py', 'biocofre/pipeline/executor.py',
            'biocofre/acesso.py', 'biocofre/banco.py', 'biocofre/cadastro.py', 'biocofre/relatorios.py',
            'biocofre/avaliacao.py', 'biocofre/experimentos.py', 'biocofre/cli.py', 'biocofre/gui.py',
            'tests/conftest.py', 'tests/test_pipeline.py', 'tests/test_banco.py', 'tests/test_acesso.py',
            'tests/test_cadastro_relatorios_avaliacao.py', 'tests/test_cli.py']
LARGURA = 100   # caracteres por linha de código (já contando o número da linha)

def descricao(caminho: Path) -> str:
    if caminho.suffix != '.py':
        return 'Dependências do projeto'
    doc = ast.get_docstring(ast.parse(caminho.read_text(encoding='utf-8')))
    return (doc or '').splitlines()[0][:78] if doc else '-'

def numerar(texto: str) -> str:
    saida = []
    for n, linha in enumerate(texto.expandtabs(4).splitlines(), 1):
        prefixo = f'{n:4d}  '
        corpo = linha.rstrip()
        espaco = LARGURA - len(prefixo)
        partes = [corpo[i:i + espaco] for i in range(0, max(len(corpo), 1), espaco)] or ['']
        saida.append(prefixo + partes[0])
        saida += [' ' * len(prefixo) + '\u21b3' + p for p in partes[1:]]
    return '\n'.join(saida)

estilo_codigo = ParagraphStyle('codigo', fontName='Mono', fontSize=6.9, leading=8.3, textColor=colors.HexColor('#111111'))
h1 = ParagraphStyle('h1', fontName='Sans-Bold', fontSize=15, leading=19, spaceAfter=8, textColor=colors.HexColor('#0b2a55'))
h2 = ParagraphStyle('h2', fontName='Sans-Bold', fontSize=10.5, leading=13, spaceBefore=4, spaceAfter=4, textColor=colors.HexColor('#0b2a55'))
corpo = ParagraphStyle('corpo', fontName='Sans', fontSize=9, leading=12.5)
celula = ParagraphStyle('celula', fontName='Sans', fontSize=7.8, leading=10)

def cabecalho_rodape(canvas, doc):
    canvas.saveState()
    canvas.setFont('Sans', 7.5)
    canvas.setFillColor(colors.HexColor('#555555'))
    canvas.drawString(2.5 * cm, A4[1] - 1.6 * cm, 'BioCofre - Relatório de código')
    canvas.drawRightString(A4[0] - 2.5 * cm, 1.2 * cm, f'Página {doc.page}')
    canvas.setStrokeColor(colors.HexColor('#cccccc'))
    canvas.line(2.5 * cm, A4[1] - 1.75 * cm, A4[0] - 2.5 * cm, A4[1] - 1.75 * cm)
    canvas.restoreState()

doc = BaseDocTemplate(str(SAIDA), pagesize=A4, leftMargin=2.5 * cm, rightMargin=2.5 * cm,
                      topMargin=2.2 * cm, bottomMargin=1.8 * cm,
                      title='BioCofre - Relatório de código', author='Grupo APS PIVC')
doc.addPageTemplates([PageTemplate(id='p', frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='f',
                                                       leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)],
                                   onPage=cabecalho_rodape)])
historia, linhas_total, linhas_por_arquivo = [], 0, []
for rel in ARQUIVOS:
    texto = (RAIZ / rel).read_text(encoding='utf-8')
    n = len(texto.splitlines())
    linhas_por_arquivo.append((rel, n, descricao(RAIZ / rel)))
    linhas_total += n

historia += [Paragraph('BioCofre', h1),
             Paragraph('Relatório com as linhas de código do programa', h2),
             Paragraph('Sistema de identificação e autenticação biométrica por reconhecimento facial com controle de acesso '
                       'em três níveis. APS de Processamento de Imagem e Visão Computacional (PIVC), UNIP, 2026/2. '
                       f'Linguagem: Python 3. Total: {len(ARQUIVOS)} arquivos e {linhas_total} linhas '
                       '(código-fonte, testes automatizados e dependências).', corpo), Spacer(1, 10)]
dados = [[Paragraph('<b>Arquivo</b>', celula), Paragraph('<b>Linhas</b>', celula), Paragraph('<b>Função</b>', celula)]]
dados += [[Paragraph(rel, celula), Paragraph(str(n), celula), Paragraph(desc, celula)] for rel, n, desc in linhas_por_arquivo]
tabela = Table(dados, colWidths=[5.6 * cm, 1.3 * cm, 9.1 * cm], repeatRows=1)
tabela.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8eef7')),
                            ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#c8d0dc')),
                            ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('TOPPADDING', (0, 0), (-1, -1), 2.5),
                            ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5)]))
historia += [tabela, PageBreak()]
for i, rel in enumerate(ARQUIVOS):
    n = linhas_por_arquivo[i][1]
    historia += [Paragraph(f'{rel}  <font size="8" color="#666666">({n} linhas)</font>', h2),
                 Preformatted(numerar((RAIZ / rel).read_text(encoding='utf-8')), estilo_codigo)]
    if i < len(ARQUIVOS) - 1:
        historia.append(PageBreak() if n > 60 else Spacer(1, 14))
doc.build(historia)
print('PDF gerado:', SAIDA, SAIDA.stat().st_size // 1024, 'KB;', linhas_total, 'linhas')
