"""Interface gráfica do BioCofre (Tkinter), com visual escuro e acentos neon.

A janela mostra o vídeo e, ao lado, a saída de cada fase do processamento: pré-processamento,
segmentação, mapa LBP e ranking de distâncias, além da decisão de acesso e do log recente.
"""
from __future__ import annotations

import time
import tkinter as tk
import webbrowser
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from . import relatorios
from .acesso import NOMES_NIVEL, Autenticador, Estado
from .banco import Banco, ErroBanco
from .cadastro import Cadastro, ErroCadastro, MENSAGENS_AMOSTRA
from .config import Config
from .pipeline.aquisicao import ErroAquisicao, abrir_fonte
from .pipeline.executor import Pipeline, ResultadoPipeline

BG, PAINEL, BORDA = "#0a0e1a", "#111827", "#22304f"
TEXTO, SUAVE = "#d7e3ff", "#8b9bbd"
CIANO, MAGENTA, VERDE, VERMELHO, AMBAR, AZUL = "#00e5ff", "#ff2bd6", "#39ff88", "#ff3b5c", "#ffb020", "#4d7cff"

COR_ESTADO = {Estado.AGUARDANDO: SUAVE, Estado.ANALISANDO: AMBAR, Estado.PIN_NECESSARIO: AZUL,
              Estado.CONCEDIDO: VERDE, Estado.NEGADO: VERMELHO, Estado.BLOQUEADO: MAGENTA}
TITULO_ESTADO = {Estado.AGUARDANDO: "AGUARDANDO", Estado.ANALISANDO: "ANALISANDO",
                 Estado.PIN_NECESSARIO: "PIN NECESSÁRIO", Estado.CONCEDIDO: "ACESSO CONCEDIDO",
                 Estado.NEGADO: "ACESSO NEGADO", Estado.BLOQUEADO: "TERMINAL BLOQUEADO"}
POSES = ["Olhe de frente", "Vire levemente a cabeça para a esquerda",
         "Vire levemente a cabeça para a direita", "Incline a cabeça um pouco para cima",
         "Sorria ou mude a expressão"]
LARGURA_VIDEO, ALTURA_VIDEO, MINI = 640, 480, 148


def _bgr(cor: str) -> tuple[int, int, int]:
    return int(cor[5:7], 16), int(cor[3:5], 16), int(cor[1:3], 16)


def _foto(rgb: np.ndarray) -> ImageTk.PhotoImage:
    return ImageTk.PhotoImage(Image.fromarray(rgb))


def _quadrado(img: np.ndarray, tam: int, nearest: bool = False) -> np.ndarray:
    """Coloca a imagem (cinza ou BGR) dentro de um quadrado ``tam`` x ``tam`` sem distorcer."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    esc = tam / max(h, w)
    interp = cv2.INTER_NEAREST if nearest else (cv2.INTER_AREA if esc < 1 else cv2.INTER_CUBIC)
    img = cv2.resize(img, (max(1, int(w * esc)), max(1, int(h * esc))), interpolation=interp)
    tela = np.full((tam, tam, 3), _bgr(PAINEL), np.uint8)
    y, x = (tam - img.shape[0]) // 2, (tam - img.shape[1]) // 2
    tela[y:y + img.shape[0], x:x + img.shape[1]] = img
    return cv2.cvtColor(tela, cv2.COLOR_BGR2RGB)


class App:
    def __init__(self, cfg: Config, origem=None, banco: Optional[Banco] = None,
                 raiz: Optional[tk.Tk] = None) -> None:
        self.cfg = cfg
        self.banco = banco or Banco(cfg.caminho_banco, cfg.caminho_chave)
        self.pipeline = Pipeline(cfg)
        self.auth = Autenticador(self.banco, cfg, self.pipeline)
        self.cadastro = Cadastro(self.banco, self.pipeline, cfg)
        self.origem = cfg.camera_indice if origem is None else origem
        self.fonte = None
        self.rodando = True
        self._captura: Optional[dict] = None
        self._pin_aberto = False
        self._ultimo_estado = Estado.AGUARDANDO
        self._fps = 0.0
        self._t_ant = time.perf_counter()
        self._admin_falhas, self._admin_bloqueado_ate = 0, 0.0
        self._fotos: dict = {}
        self._nomes: dict[int, str] = {}

        self.root = raiz or tk.Tk()
        self.root.title("BioCofre - controle de acesso por reconhecimento facial")
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self.fechar)
        self._estilo()
        self._montar()
        self._atualizar_nomes()
        self._atualizar_log()
        self._abrir_fonte()
        self.root.after(30, self._tick)

    # ---- construção da janela --------------------------------------------------------
    def _estilo(self) -> None:
        s = ttk.Style(self.root)
        s.theme_use("clam")
        fonte = ("Segoe UI", 10)
        s.configure(".", background=BG, foreground=TEXTO, font=fonte, fieldbackground=PAINEL)
        s.configure("TFrame", background=BG)
        s.configure("Painel.TFrame", background=PAINEL)
        s.configure("TLabel", background=BG, foreground=TEXTO)
        s.configure("Suave.TLabel", background=BG, foreground=SUAVE, font=("Segoe UI", 9))
        s.configure("TButton", background=PAINEL, foreground=CIANO, bordercolor=BORDA, padding=(10, 5),
                    focuscolor=PAINEL)
        s.map("TButton", background=[("active", BORDA)], foreground=[("disabled", SUAVE)])
        s.configure("TCombobox", fieldbackground=PAINEL, background=PAINEL, foreground=TEXTO,
                    arrowcolor=CIANO, bordercolor=BORDA)
        s.map("TCombobox", fieldbackground=[("readonly", PAINEL)], foreground=[("readonly", TEXTO)])
        s.configure("TEntry", fieldbackground=PAINEL, foreground=TEXTO, bordercolor=BORDA, insertcolor=CIANO)
        s.configure("Treeview", background=PAINEL, fieldbackground=PAINEL, foreground=TEXTO,
                    bordercolor=BORDA, rowheight=22)
        s.configure("Treeview.Heading", background=BORDA, foreground=CIANO, relief="flat")
        s.map("Treeview", background=[("selected", BORDA)])
        s.configure("Neon.Horizontal.TProgressbar", troughcolor=PAINEL, background=CIANO, bordercolor=BORDA)
        self.root.option_add("*TCombobox*Listbox.background", PAINEL)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXTO)

    def _montar(self) -> None:
        raiz = self.root
        barra = ttk.Frame(raiz)
        barra.pack(fill="x", padx=12, pady=(10, 6))
        ttk.Label(barra, text="BIOCOFRE", foreground=CIANO, font=("Segoe UI Semibold", 15)).pack(side="left")
        ttk.Label(barra, text="  zona:", style="Suave.TLabel").pack(side="left", padx=(16, 4))
        self._zonas = self.banco.zonas()
        self.var_zona = tk.StringVar(value=self._rotulo_zona(self._zonas[0]))
        combo = ttk.Combobox(barra, textvariable=self.var_zona, state="readonly", width=34,
                             values=[self._rotulo_zona(z) for z in self._zonas])
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", self._trocar_zona)
        for texto, comando in (("Cadastrar", self.abrir_cadastro), ("Usuários", self.abrir_usuarios),
                               ("Acessos", self.abrir_acessos), ("Relatório", self.gerar_relatorio),
                               ("Integridade", self.verificar_integridade)):
            ttk.Button(barra, text=texto, command=comando).pack(side="left", padx=(8, 0))
        self.btn_pausa = ttk.Button(barra, text="Pausar", command=self.alternar)
        self.btn_pausa.pack(side="right")

        corpo = ttk.Frame(raiz)
        corpo.pack(fill="both", expand=True, padx=12)
        esq = ttk.Frame(corpo)
        esq.pack(side="left", anchor="n")
        self.cv_video = tk.Canvas(esq, width=LARGURA_VIDEO, height=ALTURA_VIDEO, bg="#000000",
                                  highlightthickness=1, highlightbackground=BORDA)
        self.cv_video.pack()
        self._item_video = self.cv_video.create_image(0, 0, anchor="nw")
        self.lbl_f1 = ttk.Label(esq, text="1 · Aquisição", style="Suave.TLabel")
        self.lbl_f1.pack(anchor="w", pady=(4, 0))

        dir_ = ttk.Frame(corpo)
        dir_.pack(side="left", anchor="n", padx=(14, 0))
        ttk.Label(dir_, text="Fases do processamento", foreground=CIANO,
                  font=("Segoe UI Semibold", 11)).pack(anchor="w")
        linha = ttk.Frame(dir_)
        linha.pack(anchor="w", pady=(4, 0))
        self._minis = {}
        for chave, titulo in (("pre", "2 · Pré-processamento"), ("seg", "3 · Segmentação"),
                              ("lbp", "4 · Extração (LBP)")):
            col = ttk.Frame(linha)
            col.pack(side="left", padx=(0, 8))
            rotulo = tk.Label(col, bg=PAINEL, width=MINI, height=MINI, bd=0, highlightthickness=1,
                              highlightbackground=BORDA)
            rotulo.configure(image=_foto(_quadrado(np.zeros((8, 8), np.uint8), MINI)))
            rotulo.pack()
            self._minis[chave] = rotulo
            ttk.Label(col, text=titulo, style="Suave.TLabel").pack(anchor="w")
        ttk.Label(dir_, text="5 · Classificação (menor distância por usuário)",
                  style="Suave.TLabel").pack(anchor="w", pady=(8, 2))
        self.cv_rank = tk.Canvas(dir_, width=474, height=118, bg=PAINEL, highlightthickness=1,
                                 highlightbackground=BORDA)
        self.cv_rank.pack(anchor="w")
        self.lbl_estado = tk.Label(dir_, text="AGUARDANDO", bg=PAINEL, fg=SUAVE, width=26,
                                   font=("Segoe UI Semibold", 20), pady=10, highlightthickness=2,
                                   highlightbackground=SUAVE)
        self.lbl_estado.pack(anchor="w", fill="x", pady=(10, 0))
        self.lbl_msg = ttk.Label(dir_, text="", wraplength=470, font=("Segoe UI", 10))
        self.lbl_msg.pack(anchor="w", pady=(6, 2))
        self.barra = ttk.Progressbar(dir_, style="Neon.Horizontal.TProgressbar", maximum=1.0, length=474)
        self.barra.pack(anchor="w")
        self.lbl_info = ttk.Label(dir_, text="", style="Suave.TLabel", wraplength=470, justify="left")
        self.lbl_info.pack(anchor="w", pady=(6, 0))
        self.lbl_tempos = ttk.Label(dir_, text="", style="Suave.TLabel")
        self.lbl_tempos.pack(anchor="w")

        ttk.Label(raiz, text="Últimos eventos", style="Suave.TLabel").pack(anchor="w", padx=12, pady=(8, 0))
        colunas = ("hora", "evento", "zona", "usuario", "motivo", "distancia")
        self.tv_log = ttk.Treeview(raiz, columns=colunas, show="headings", height=5)
        for c, titulo, larg in (("hora", "Hora", 150), ("evento", "Evento", 90), ("zona", "Zona", 250),
                                ("usuario", "Usuário", 220), ("motivo", "Motivo", 340),
                                ("distancia", "Distância", 90)):
            self.tv_log.heading(c, text=titulo)
            self.tv_log.column(c, width=larg, anchor="w")
        self.tv_log.pack(fill="x", padx=12, pady=(2, 10))

    # ---- utilidades -----------------------------------------------------------------
    @staticmethod
    def _rotulo_zona(zona) -> str:
        return f"{zona.nome} (Nível {zona.nivel})"

    def _trocar_zona(self, _evento=None) -> None:
        indice = [self._rotulo_zona(z) for z in self._zonas].index(self.var_zona.get())
        self.auth.definir_zona(self._zonas[indice])

    def _atualizar_nomes(self) -> None:
        self._nomes = {u.id: u.nome for u in self.banco.listar_usuarios()}

    def _atualizar_log(self) -> None:
        self.tv_log.delete(*self.tv_log.get_children())
        for r in self.banco.listar_acessos(limite=5):
            dist = "" if r["distancia"] is None else f"{r['distancia']:.3f}"
            self.tv_log.insert("", "end", values=(r["ts"].replace("T", " "), r["evento"], r["zona"] or "-",
                                                  r["usuario_nome"] or "-", r["motivo"] or "", dist))

    def _abrir_fonte(self) -> None:
        try:
            self.fonte = abrir_fonte(self.origem, self.cfg)
        except ErroAquisicao as exc:
            self.fonte = None
            self.cv_video.delete("aviso")
            self.cv_video.create_text(LARGURA_VIDEO // 2, ALTURA_VIDEO // 2, text=str(exc), fill=VERMELHO,
                                      width=520, font=("Segoe UI", 12), tags="aviso")

    def alternar(self) -> None:
        self.rodando = not self.rodando
        self.btn_pausa.configure(text="Continuar" if not self.rodando else "Pausar")

    # ---- laço principal ---------------------------------------------------------------
    def _tick(self) -> None:
        try:
            if self.rodando and self.fonte is not None:
                quadro = self.fonte.ler()
                if quadro is not None:
                    if self._captura is not None:
                        self._passo_captura(quadro)
                    else:
                        situacao, res = self.auth.processar(quadro)
                        self._atualizar(quadro, situacao, res)
                    agora = time.perf_counter()
                    self._fps = 0.9 * self._fps + 0.1 / max(agora - self._t_ant, 1e-6) if self._fps else 1 / max(agora - self._t_ant, 1e-6)
                    self._t_ant = agora
        except Exception as exc:  # a interface não pode parar por um quadro problemático
            self.lbl_msg.configure(text=f"Erro ao processar o quadro: {exc}")
        self.root.after(15, self._tick)

    def _atualizar(self, quadro: np.ndarray, situacao, res: Optional[ResultadoPipeline]) -> None:
        cor = COR_ESTADO[situacao.estado]
        caixa = res.seg.caixa if res is not None and res.seg is not None else None
        self._mostrar_video(quadro, caixa, cor)
        if res is not None:
            self._mostrar_fases(res)
            self._desenhar_ranking(res.ranking, self.cfg.limiar_do_nivel(self.auth.zona.nivel))
            t = res.tempos_ms
            self.lbl_tempos.configure(text=(
                f"pré {t.get('preprocessamento', 0):.1f} ms · seg {t.get('segmentacao', 0):.1f} ms · "
                f"LBP {t.get('caracteristicas', 0):.1f} ms · class {t.get('classificacao', 0):.1f} ms · "
                f"{self._fps:.0f} quadros/s"))
        self.lbl_estado.configure(text=TITULO_ESTADO[situacao.estado], fg=cor, highlightbackground=cor)
        self.lbl_msg.configure(text=situacao.mensagem if situacao.estado != Estado.BLOQUEADO else
                               f"{situacao.mensagem} ({situacao.bloqueio_restante_s} s)")
        self.barra["value"] = situacao.progresso
        zona = self.auth.zona
        info = f"Zona: {zona.nome} · limiar {self.cfg.limiar_do_nivel(zona.nivel):.3f} · " \
               f"confirmação em {self.cfg.quadros_confirmacao[zona.nivel]} quadros"
        if situacao.usuario is not None:
            u = situacao.usuario
            info = (f"{u.nome} · {NOMES_NIVEL[u.nivel]}" + (f" · divisão {u.divisao}" if u.divisao else "")
                    + (f" · distância {situacao.distancia:.3f}" if situacao.distancia is not None else "") + "\n" + info)
        self.lbl_info.configure(text=info)
        self.lbl_f1.configure(text=f"1 · Aquisição: {quadro.shape[1]}x{quadro.shape[0]} pixels")
        if situacao.estado != self._ultimo_estado:
            if situacao.estado in (Estado.CONCEDIDO, Estado.NEGADO, Estado.BLOQUEADO):
                self._atualizar_log()
            if situacao.estado == Estado.PIN_NECESSARIO and not self._pin_aberto:
                self._ultimo_estado = situacao.estado
                self._pedir_pin()
            self._ultimo_estado = situacao.estado

    def _mostrar_video(self, quadro: np.ndarray, caixa, cor: str) -> None:
        img = quadro.copy()
        if caixa is not None:
            x, y, w, h = caixa
            cv2.rectangle(img, (x, y), (x + w, y + h), _bgr(cor), max(2, quadro.shape[1] // 200))
        h0, w0 = img.shape[:2]
        esc = min(LARGURA_VIDEO / w0, ALTURA_VIDEO / h0)
        novo = (max(1, int(w0 * esc)), max(1, int(h0 * esc)))
        img = cv2.resize(img, novo, interpolation=cv2.INTER_AREA if esc < 1 else cv2.INTER_LINEAR)
        foto = _foto(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.cv_video.itemconfig(self._item_video, image=foto)
        self.cv_video.coords(self._item_video, (LARGURA_VIDEO - novo[0]) // 2, (ALTURA_VIDEO - novo[1]) // 2)
        self._fotos["video"] = foto

    def _mostrar_fases(self, res: ResultadoPipeline) -> None:
        imagens = {}
        if res.pre is not None:
            imagens["pre"] = _quadrado(res.pre.suavizada, MINI)
        if res.seg is not None and res.seg.recorte is not None:
            imagens["seg"] = _quadrado(res.seg.recorte, MINI)
        if res.lbp is not None:
            grade = self.cfg.lbp_grade
            lbp = _quadrado(res.lbp, MINI, nearest=True)
            for i in range(1, grade):
                p = int(i * MINI / grade)
                cv2.line(lbp, (p, 0), (p, MINI), (0, 229, 255), 1)
                cv2.line(lbp, (0, p), (MINI, p), (0, 229, 255), 1)
            imagens["lbp"] = lbp
        for chave, rgb in imagens.items():
            foto = _foto(rgb)
            self._minis[chave].configure(image=foto)
            self._fotos[chave] = foto

    def _desenhar_ranking(self, ranking, limiar: float) -> None:
        c = self.cv_rank
        c.delete("all")
        if not ranking:
            c.create_text(237, 59, text="sem comparação", fill=SUAVE)
            return
        topo = ranking[:5]
        maior = max(limiar * 1.5, max(r.distancia for r in topo))
        esc = (474 - 206) / maior
        for i, cand in enumerate(topo):
            y = 8 + i * 21
            nome = self._nomes.get(cand.usuario_id, f"#{cand.usuario_id}")
            cor = VERDE if cand.distancia <= limiar else "#4b5a7a"
            c.create_text(6, y + 8, anchor="w", text=nome[:26], fill=TEXTO, font=("Segoe UI", 9))
            c.create_rectangle(176, y, 176 + cand.distancia * esc, y + 16, fill=cor, outline="")
            c.create_text(180 + cand.distancia * esc, y + 8, anchor="w", text=f"{cand.distancia:.3f}",
                          fill=TEXTO, font=("Consolas", 9))
        x = 176 + limiar * esc
        c.create_line(x, 0, x, 118, fill=MAGENTA, dash=(4, 3))
        c.create_text(x - 4, 112, anchor="e", text=f"limiar {limiar:.3f}", fill=MAGENTA, font=("Segoe UI", 8))

    # ---- PIN, administrador --------------------------------------------------------------
    def _pedir_pin(self) -> None:
        self._pin_aberto = True
        try:
            pin = simpledialog.askstring("Segundo fator", f"{self.auth.situacao.mensagem}", show="*",
                                         parent=self.root)
            situacao = self.auth.cancelar_pin() if pin is None else self.auth.informar_pin(pin)
        finally:
            self._pin_aberto = False
        self._atualizar_apos_pin(situacao)

    def _atualizar_apos_pin(self, situacao) -> None:
        self._ultimo_estado = situacao.estado
        self.lbl_estado.configure(text=TITULO_ESTADO[situacao.estado], fg=COR_ESTADO[situacao.estado],
                                  highlightbackground=COR_ESTADO[situacao.estado])
        self.lbl_msg.configure(text=situacao.mensagem)
        self._atualizar_log()
        if situacao.estado == Estado.PIN_NECESSARIO:  # PIN incorreto, ainda há tentativas
            self.root.after(50, self._pedir_pin)

    def _autenticar_admin(self) -> bool:
        if time.monotonic() < self._admin_bloqueado_ate:
            messagebox.showerror("Administrador", "Muitas tentativas incorretas. Aguarde um minuto.")
            return False
        if not self.banco.tem_senha_admin():
            s1 = simpledialog.askstring("Administrador", "Primeiro uso: defina a senha do administrador "
                                        "(mínimo 6 caracteres):", show="*", parent=self.root)
            if not s1:
                return False
            if s1 != simpledialog.askstring("Administrador", "Repita a senha:", show="*", parent=self.root):
                messagebox.showerror("Administrador", "As senhas não conferem.")
                return False
            try:
                self.banco.definir_senha_admin(s1)
            except ErroBanco as exc:
                messagebox.showerror("Administrador", str(exc))
                return False
            return True
        senha = simpledialog.askstring("Administrador", "Senha do administrador:", show="*", parent=self.root)
        if senha and self.banco.verificar_senha_admin(senha):
            self._admin_falhas = 0
            return True
        if senha is not None:
            self._admin_falhas += 1
            if self._admin_falhas >= 5:
                self._admin_bloqueado_ate, self._admin_falhas = time.monotonic() + 60, 0
            messagebox.showerror("Administrador", "Senha incorreta.")
        return False

    # ---- cadastro ------------------------------------------------------------------------
    def abrir_cadastro(self) -> None:
        if self._captura is None and self._autenticar_admin():
            JanelaCadastro(self)

    def iniciar_captura(self, dados: dict, janela: "JanelaCadastro") -> None:
        self._captura = {"dados": dados, "amostras": [], "ultimo": 0.0, "janela": janela}
        self.rodando = True

    def cancelar_captura(self) -> None:
        self._captura = None

    def _passo_captura(self, quadro: np.ndarray) -> None:
        cap = self._captura
        amostra = self.cadastro.avaliar_quadro(quadro)
        res = amostra.resultado
        caixa = res.seg.caixa if res is not None and res.seg is not None else None
        self._mostrar_video(quadro, caixa, VERDE if amostra.aceita else AMBAR)
        if res is not None:
            self._mostrar_fases(res)
        agora = time.monotonic()
        if amostra.aceita and agora - cap["ultimo"] >= 0.35:
            cap["amostras"].append(amostra)
            cap["ultimo"] = agora
        n, alvo = len(cap["amostras"]), cap["dados"]["amostras"]
        pose = POSES[min(len(POSES) - 1, n * len(POSES) // alvo)]
        self.lbl_estado.configure(text=f"CADASTRO {n}/{alvo}", fg=CIANO, highlightbackground=CIANO)
        self.lbl_msg.configure(text=f"{pose}\n{MENSAGENS_AMOSTRA.get(amostra.motivo, amostra.motivo)}")
        self.barra["value"] = n / alvo
        cap["janela"].atualizar(n, alvo)
        if n >= alvo:
            self._finalizar_captura()

    def _finalizar_captura(self) -> None:
        cap, self._captura = self._captura, None
        d = cap["dados"]
        try:
            usuario = self.cadastro.cadastrar(d["nome"], d["cargo"], d["nivel"], d["divisao"], d["pin"],
                                              cap["amostras"], self.auth.galeria)
        except (ErroCadastro, ErroBanco) as exc:
            messagebox.showerror("Cadastro não realizado", str(exc), parent=cap["janela"])
        else:
            messagebox.showinfo("Cadastro", f"{usuario.nome} cadastrado como {NOMES_NIVEL[usuario.nivel]}.",
                                parent=cap["janela"])
            self.auth.recarregar_galeria()
            self._atualizar_nomes()
            self._atualizar_log()
        cap["janela"].destroy()

    # ---- usuários, acessos, relatórios -----------------------------------------------------
    def abrir_usuarios(self) -> None:
        if self._autenticar_admin():
            JanelaUsuarios(self)

    def abrir_acessos(self) -> None:
        JanelaAcessos(self)

    def gerar_relatorio(self) -> None:
        pasta = Path(self.cfg.pasta_dados) / "relatorios"
        carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
        registros = self.banco.listar_acessos(limite=0)
        relatorios.exportar_csv(registros, pasta / f"acessos_{carimbo}.csv")
        caminho = relatorios.gerar_html(self.banco, pasta / f"relatorio_{carimbo}.html")
        webbrowser.open(caminho.resolve().as_uri())
        messagebox.showinfo("Relatório", f"Relatório gerado em:\n{caminho}")

    def verificar_integridade(self) -> None:
        r = self.banco.verificar_integridade()
        (messagebox.showinfo if r.ok else messagebox.showerror)("Integridade do log", r.mensagem)

    def fechar(self) -> None:
        self.rodando = False
        if self.fonte is not None:
            self.fonte.fechar()
        self.banco.fechar()
        self.root.destroy()


class JanelaCadastro(tk.Toplevel):
    def __init__(self, app: App) -> None:
        super().__init__(app.root)
        self.app = app
        self.title("Cadastrar usuário")
        self.configure(bg=BG, padx=16, pady=14)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._fechar)
        self.v = {k: tk.StringVar() for k in ("nome", "cargo", "nivel", "divisao", "pin")}
        self.v["nivel"].set("1")
        self.v["amostras"] = tk.StringVar(value=str(app.cfg.amostras_cadastro))
        divisoes = sorted({z.divisao for z in app.banco.zonas() if z.divisao})
        linhas = [("Nome completo", ttk.Entry(self, textvariable=self.v["nome"], width=36)),
                  ("Cargo", ttk.Entry(self, textvariable=self.v["cargo"], width=36)),
                  ("Nível (1, 2 ou 3)", ttk.Combobox(self, textvariable=self.v["nivel"], values=("1", "2", "3"),
                                                     state="readonly", width=6)),
                  ("Divisão (Nível 2)", ttk.Combobox(self, textvariable=self.v["divisao"], values=divisoes, width=34)),
                  ("PIN (4 a 8 dígitos; obrigatório no Nível 3)", ttk.Entry(self, textvariable=self.v["pin"], show="*", width=12)),
                  ("Amostras", ttk.Spinbox(self, from_=5, to=40, textvariable=self.v["amostras"], width=6))]
        for i, (rotulo, widget) in enumerate(linhas):
            ttk.Label(self, text=rotulo).grid(row=i, column=0, sticky="w", pady=3, padx=(0, 12))
            widget.grid(row=i, column=1, sticky="w", pady=3)
        self.lbl = ttk.Label(self, text="Preencha os dados e inicie a captura.", style="Suave.TLabel", wraplength=420)
        self.lbl.grid(row=len(linhas), column=0, columnspan=2, sticky="w", pady=(10, 4))
        self.barra = ttk.Progressbar(self, style="Neon.Horizontal.TProgressbar", maximum=1.0, length=420)
        self.barra.grid(row=len(linhas) + 1, column=0, columnspan=2, pady=(0, 8))
        self.btn = ttk.Button(self, text="Iniciar captura", command=self._iniciar)
        self.btn.grid(row=len(linhas) + 2, column=0, columnspan=2)

    def _iniciar(self) -> None:
        try:
            dados = {"nome": self.v["nome"].get().strip(), "cargo": self.v["cargo"].get().strip(),
                     "nivel": int(self.v["nivel"].get()), "divisao": self.v["divisao"].get().strip() or None,
                     "pin": self.v["pin"].get().strip() or None, "amostras": int(self.v["amostras"].get())}
        except ValueError:
            messagebox.showerror("Cadastro", "Confira o nível e o número de amostras.", parent=self)
            return
        if not dados["nome"]:
            messagebox.showerror("Cadastro", "Informe o nome.", parent=self)
            return
        self.btn.configure(state="disabled")
        self.lbl.configure(text="Olhe para a câmera. A captura começa agora.")
        self.app.iniciar_captura(dados, self)

    def atualizar(self, n: int, alvo: int) -> None:
        self.barra["value"] = n / alvo
        self.lbl.configure(text=f"{n} de {alvo} amostras de boa qualidade")

    def _fechar(self) -> None:
        self.app.cancelar_captura()
        self.destroy()


class JanelaUsuarios(tk.Toplevel):
    def __init__(self, app: App) -> None:
        super().__init__(app.root)
        self.app = app
        self.title("Usuários cadastrados")
        self.configure(bg=BG, padx=12, pady=12)
        cols = ("id", "nome", "cargo", "nivel", "divisao", "pin", "estado")
        self.tv = ttk.Treeview(self, columns=cols, show="headings", height=12)
        for c, t, w in (("id", "ID", 40), ("nome", "Nome", 220), ("cargo", "Cargo", 190), ("nivel", "Nível", 50),
                        ("divisao", "Divisão", 120), ("pin", "PIN", 50), ("estado", "Estado", 90)):
            self.tv.heading(c, text=t)
            self.tv.column(c, width=w, anchor="w")
        self.tv.pack()
        botoes = ttk.Frame(self)
        botoes.pack(fill="x", pady=(8, 0))
        ttk.Button(botoes, text="Ativar/Desativar", command=self._alternar).pack(side="left")
        ttk.Button(botoes, text="Remover", command=self._remover).pack(side="left", padx=8)
        self._carregar()

    def _carregar(self) -> None:
        self.tv.delete(*self.tv.get_children())
        for u in self.app.banco.listar_usuarios():
            self.tv.insert("", "end", iid=str(u.id), values=(u.id, u.nome, u.cargo, u.nivel, u.divisao or "-",
                                                             "sim" if u.tem_pin else "não",
                                                             "ativo" if u.ativo else "desativado"))

    def _selecionado(self):
        sel = self.tv.selection()
        return self.app.banco.obter_usuario(int(sel[0])) if sel else None

    def _depois(self) -> None:
        self.app.auth.recarregar_galeria()
        self.app._atualizar_nomes()
        self.app._atualizar_log()
        self._carregar()

    def _alternar(self) -> None:
        u = self._selecionado()
        if u:
            self.app.banco.definir_ativo(u.id, not u.ativo)
            self.app.banco.registrar_acesso("ADMIN", None, u.id, u.nome, "ativado" if not u.ativo else "desativado")
            self._depois()

    def _remover(self) -> None:
        u = self._selecionado()
        if u and messagebox.askyesno("Remover", f"Remover '{u.nome}' e todos os seus modelos?", parent=self):
            self.app.banco.remover_usuario(u.id)
            self.app.banco.registrar_acesso("REMOCAO", None, None, u.nome, "removido_pelo_administrador")
            self._depois()


class JanelaAcessos(tk.Toplevel):
    def __init__(self, app: App) -> None:
        super().__init__(app.root)
        self.app = app
        self.title("Registro de acessos")
        self.configure(bg=BG, padx=12, pady=12)
        cols = ("id", "ts", "evento", "zona", "usuario", "motivo", "distancia")
        self.tv = ttk.Treeview(self, columns=cols, show="headings", height=16)
        for c, t, w in (("id", "ID", 50), ("ts", "Data e hora", 140), ("evento", "Evento", 100), ("zona", "Zona", 230),
                        ("usuario", "Usuário", 190), ("motivo", "Motivo", 280), ("distancia", "Distância", 80)):
            self.tv.heading(c, text=t)
            self.tv.column(c, width=w, anchor="w")
        self.tv.pack()
        for r in app.banco.listar_acessos(limite=300):
            dist = "" if r["distancia"] is None else f"{r['distancia']:.3f}"
            self.tv.insert("", "end", values=(r["id"], r["ts"].replace("T", " "), r["evento"], r["zona"] or "-",
                                              r["usuario_nome"] or "-", r["motivo"] or "", dist))
        ttk.Button(self, text="Verificar integridade", command=app.verificar_integridade).pack(anchor="w", pady=(8, 0))


def executar_gui(cfg: Config, origem=None) -> None:
    if origem is not None and str(origem).isdigit():
        origem = int(origem)
    App(cfg, origem).root.mainloop()
