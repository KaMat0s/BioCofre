# BioCofre

Sistema de identificação e autenticação biométrica por **reconhecimento facial em tempo real** para
controlar o acesso a um cofre com **três níveis de segurança** (APS de Processamento de Imagem e
Visão Computacional, UNIP, 2026/2).

| Nível | Quem entra |
| --- | --- |
| 1 | Usuários com permissão geral |
| 2 | Diretores das divisões específicas (cada zona de Nível 2 pertence a uma divisão); o ministro também entra |
| 3 | Exclusivamente o ministro, com rosto **e PIN** |

O programa segue as cinco fases do processamento de imagens: **aquisição → pré-processamento →
segmentação → extração de características → classificação/interpretação**. A interface mostra a
saída de cada fase em tempo real.

## Instalação

Requer Python 3.10 ou mais novo.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
# Linux: se faltar o Tkinter  ->  sudo apt install python3-tk
python main.py            # abre a interface, usando a câmera 0
```

No primeiro uso, ao clicar em **Cadastrar**, o programa pede para definir a senha do administrador.
Para cadastrar: preencha nome, nível (e divisão no Nível 2; PIN no Nível 3), clique em *Iniciar captura*
e siga as instruções de pose (15 amostras por padrão). Os dados ficam na pasta `dados/`
(banco `biocofre.db` e chave `chave_log.key`).

## Uso pela linha de comando

| Comando | O que faz |
| --- | --- |
| `python main.py gui [--origem 0\|pasta\|arquivo]` | interface gráfica (câmera, pasta de imagens ou vídeo) |
| `python main.py cadastrar --nome "Ana" --nivel 1` | cadastra capturando da câmera |
| `python main.py usuarios` / `remover --id N` | lista / remove usuários |
| `python main.py acessos --limite 30` | últimas decisões do log |
| `python main.py integridade` | verifica a cadeia de HMAC do log |
| `python main.py relatorio --saida relatorios` | gera CSV e HTML dos acessos |
| `python main.py identificar --imagem foto.jpg --zona 4` | diagnóstico: roda uma imagem pelas cinco fases |
| `python main.py avaliar --base pasta_por_pessoa` | identificação, EER e limiares sugeridos |
| `python main.py calibrar --base minhas_fotos --gravar config.json` | calibra os limiares com fotos do seu ambiente |
| `python main.py experimentos --base orl --saida resultados` | reproduz os experimentos da dissertação |
| `python main.py demo --base orl` | cria usuários e um roteiro de quadros para testar **sem câmera** |

## Estrutura

```
main.py                      ponto de entrada
biocofre/
  config.py                  todos os parâmetros (JSON)
  pipeline/
    aquisicao.py             fase 1  câmera, vídeo, arquivos
    preprocessamento.py      fase 2  cinza, filtro gaussiano, CLAHE, qualidade
    segmentacao.py           fase 3  detector Haar, recorte, máscara elíptica
    caracteristicas.py       fase 4  LBP uniforme, histogramas por região
    classificacao.py         fase 5  galeria, distância qui-quadrado, decisão
    executor.py              orquestra as fases e mede o tempo de cada uma
  acesso.py                  níveis, zonas, confirmação, PIN, bloqueio
  banco.py                   SQLite, PIN com PBKDF2, log encadeado por HMAC
  cadastro.py                qualidade das amostras, duplicidade, gravação
  relatorios.py              resumo, CSV, HTML
  avaliacao.py               identificação, FAR/FRR/EER, conjunto aberto, calibração
  experimentos.py            protocolo completo da dissertação
  cli.py  gui.py             linha de comando e interface Tkinter
tests/                       55 testes (pytest)
ferramentas/                 gerar_relatorio_codigo.py: PDF com todas as linhas de código (pip install reportlab)
resultados/                  resumo.json e gráficos dos experimentos
```

## Como a decisão é tomada

1. Cada quadro passa pelas cinco fases e gera um ranking de distâncias por usuário.
2. O melhor candidato só vale se a distância for menor que o **limiar do nível da zona**
   (0,422 / 0,396 / 0,386) e se `d1/d2 <= 0,90` (evita confusão entre dois cadastrados).
3. A identidade precisa se repetir em **3 / 5 / 8 quadros seguidos** (Níveis 1 / 2 / 3).
4. O nível e a divisão do usuário são conferidos contra a zona. No Nível 3 pede-se o PIN (3 tentativas).
5. Três negativas seguidas bloqueiam o terminal por 60 s. Toda decisão vai para o log encadeado.

## Calibração e avaliação

Os limiares foram calibrados com a base ORL (pessoas 1 a 20) e medidos em pessoas que o sistema não
viu (21 a 40). **Esses valores dependem da câmera e da iluminação.** Antes de usar de verdade, tire
8 ou mais fotos de pelo menos 3 pessoas, uma pasta por pessoa, e rode `calibrar`.

A base ORL (AT&T Laboratories Cambridge, 40 pessoas x 10 imagens) deve ser baixada do site oficial:
<https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html>. Ao usá-la, dê crédito ao
Olivetti Research Laboratory. Ela fica organizada como `orl/s1/1.pgm ... orl/s40/10.pgm`.

## Limitações conhecidas

* Não há detecção de vivacidade: uma foto ou um vídeo diante da câmera pode enganar o rosto. Por isso o
  Nível 3 exige PIN, mas o sistema **não deve proteger algo real** sem esse recurso.
* O LBP é sensível a pose, desalinhamento e desfoque; redes neurais são mais robustas.
* Os modelos (histogramas) são dados pessoais sensíveis (LGPD, art. 5º, II) e ficam sem criptografia no banco.
* A taxa de falsa identificação cresce com o número de cadastrados: recalibre se a galeria passar de ~15 pessoas.

## Testes

```bash
pip install pytest
python -m pytest -q                                   # testes unitários
BIOCOFRE_ORL=/caminho/orl python -m pytest -q         # inclui o teste de integração com a ORL
```
