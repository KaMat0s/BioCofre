"""Política de acesso: níveis, zonas, autorização, confirmação em quadros consecutivos e bloqueio.

O ``Autenticador`` recebe quadros de vídeo, aplica o pipeline de cinco fases e decide:

* a identidade só é aceita depois de N quadros consecutivos com o mesmo usuário, e N cresce
  com o nível da zona (limiar de distância e razão d1/d2 também);
* o usuário precisa ter nível suficiente (e, no Nível 2, ser diretor da divisão da zona);
* zonas de Nível 3 exigem um PIN além do rosto;
* tentativas negadas seguidas bloqueiam o terminal por um tempo;
* toda decisão vai para o log de auditoria encadeado.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import numpy as np

from .banco import Banco, Usuario, Zona
from .config import Config
from .pipeline.caracteristicas import N_UNIFORME
from .pipeline.classificacao import Galeria, decidir
from .pipeline.executor import Pipeline, ResultadoPipeline

NOMES_NIVEL = {1: "Nível 1 (permissão geral)", 2: "Nível 2 (diretor de divisão)", 3: "Nível 3 (ministro)"}

MENSAGENS = {
    "face_nao_reconhecida": "Rosto não reconhecido",
    "tempo_esgotado": "Não foi possível confirmar a identidade a tempo",
    "pin_incorreto": "PIN incorreto",
    "pin_expirado": "Tempo para informar o PIN esgotado",
    "pin_cancelado": "Entrada do PIN cancelada",
    "pin_nao_cadastrado": "Usuário sem PIN cadastrado para esta zona",
    "nao_autorizado:nivel_insuficiente": "Nível de acesso insuficiente para esta zona",
    "nao_autorizado:divisao_diferente": "Zona restrita aos diretores de outra divisão",
    "nao_autorizado:usuario_inativo": "Usuário desativado",
}


class Estado(str, Enum):
    AGUARDANDO = "aguardando"
    ANALISANDO = "analisando"
    PIN_NECESSARIO = "pin_necessario"
    CONCEDIDO = "concedido"
    NEGADO = "negado"
    BLOQUEADO = "bloqueado"


@dataclass
class Situacao:
    estado: Estado
    mensagem: str = ""
    usuario: Optional[Usuario] = None
    distancia: Optional[float] = None
    progresso: float = 0.0
    bloqueio_restante_s: int = 0


def autorizado(usuario: Usuario, zona: Zona) -> tuple[bool, str]:
    """Regra dos três níveis.

    Nível 1: qualquer usuário ativo. Nível 2: diretores da divisão da zona (o ministro também
    entra). Nível 3: somente o ministro.
    """
    if not usuario.ativo:
        return False, "usuario_inativo"
    if usuario.nivel < zona.nivel:
        return False, "nivel_insuficiente"
    if zona.nivel == 2 and usuario.nivel == 2 and zona.divisao and usuario.divisao != zona.divisao:
        return False, "divisao_diferente"
    return True, "ok"


def montar_galeria(banco: Banco, pipeline: Pipeline) -> Galeria:
    galeria = Galeria(pipeline.cfg.lbp_grade, N_UNIFORME, pipeline.pesos)
    modelos = banco.carregar_modelos()
    if modelos:
        galeria.adicionar_varios([m[0] for m in modelos], [m[1] for m in modelos])
    return galeria


class Autenticador:
    def __init__(self, banco: Banco, cfg: Config, pipeline: Optional[Pipeline] = None,
                 relogio: Callable[[], float] = time.monotonic) -> None:
        self.banco = banco
        self.cfg = cfg
        self.pipeline = pipeline or Pipeline(cfg)
        self._relogio = relogio
        self.galeria = montar_galeria(banco, self.pipeline)
        self.zona: Zona = banco.zonas()[0]
        self._falhas = 0
        self._bloqueado_ate = 0.0
        self._exibir_ate = 0.0
        self._historico: deque = deque(
            maxlen=max(max(cfg.quadros_confirmacao.values()), cfg.quadros_negacao))
        self._armado = True
        self._sem_rosto_desde: Optional[float] = None
        self._reiniciar_sessao()

    # ---- controle -------------------------------------------------------------------
    def recarregar_galeria(self) -> None:
        self.galeria = montar_galeria(self.banco, self.pipeline)
        self._reiniciar_sessao()

    def definir_zona(self, zona: Zona) -> None:
        self.zona = zona
        self._reiniciar_sessao()

    def _reiniciar_sessao(self) -> None:
        self._historico.clear()
        self._inicio_analise: Optional[float] = None
        self._ultimo_rosto: Optional[float] = None
        self._usuario_pin: Optional[Usuario] = None
        self._dist_pin: Optional[float] = None
        self._pin_tentativas = 0
        self._pin_ate = 0.0
        self._situacao = Situacao(Estado.AGUARDANDO, "Aproxime o rosto da câmera")

    @property
    def situacao(self) -> Situacao:
        return self._situacao

    # ---- processamento de quadros ----------------------------------------------------
    def processar(self, quadro: np.ndarray, ja_recortado: bool = False
                  ) -> tuple[Situacao, Optional[ResultadoPipeline]]:
        agora = self._relogio()
        if agora < self._bloqueado_ate:
            self._situacao = Situacao(Estado.BLOQUEADO, "Terminal bloqueado após tentativas negadas",
                                      bloqueio_restante_s=int(self._bloqueado_ate - agora) + 1)
            return self._situacao, None
        if self._situacao.estado == Estado.BLOQUEADO:
            self._reiniciar_sessao()
        if self._situacao.estado in (Estado.CONCEDIDO, Estado.NEGADO):
            if agora < self._exibir_ate:
                return self._situacao, None
            self._reiniciar_sessao()
            self._armado, self._sem_rosto_desde = False, None
        if self._situacao.estado == Estado.PIN_NECESSARIO:
            if agora > self._pin_ate:
                return self._negar("pin_expirado", self._usuario_pin, agora, self._dist_pin), None
            return self._situacao, None

        res = self.pipeline.processar(quadro, self.galeria, ja_recortado)
        tem_rosto = res.problema != "sem_rosto"

        if not self._armado:  # depois de uma decisão, o rosto precisa sair do quadro
            if tem_rosto:
                self._sem_rosto_desde = None
                self._situacao = Situacao(Estado.AGUARDANDO, "Afaste-se para uma nova leitura")
            else:
                if self._sem_rosto_desde is None:
                    self._sem_rosto_desde = agora
                if agora - self._sem_rosto_desde >= self.cfg.rearme_sem_rosto_s:
                    self._armado = True
                    self._reiniciar_sessao()
            return self._situacao, res

        if not tem_rosto:
            if self._ultimo_rosto is not None and agora - self._ultimo_rosto > self.cfg.sem_rosto_reinicia_s:
                self._reiniciar_sessao()
            self._situacao = Situacao(Estado.AGUARDANDO, "Aproxime o rosto da câmera")
            return self._situacao, res

        self._ultimo_rosto = agora
        if self._inicio_analise is None:
            self._inicio_analise = agora
        if res.problema == "varios_rostos":
            self._historico.clear()
            self._situacao = Situacao(Estado.ANALISANDO, "Apenas uma pessoa por vez no quadro")
            return self._situacao, res

        nivel = self.zona.nivel
        dec = decidir(res.ranking, self.cfg.limiar_do_nivel(nivel), self.cfg.razao_max)
        self._historico.append(dec)
        n_ok = self.cfg.quadros_confirmacao[nivel]
        ultimos = list(self._historico)[-n_ok:]
        if (len(ultimos) == n_ok and all(d.aceito for d in ultimos)
                and len({d.usuario_id for d in ultimos}) == 1):
            media = float(np.mean([d.distancia for d in ultimos]))
            return self._autorizar(ultimos[-1].usuario_id, media, agora), res

        if agora - self._inicio_analise > self.cfg.tempo_max_analise_s:
            return self._negar("tempo_esgotado", None, agora, dec.distancia), res
        n_neg = self.cfg.quadros_negacao
        if len(self._historico) >= n_neg and not any(d.aceito for d in list(self._historico)[-n_neg:]):
            return self._negar("face_nao_reconhecida", None, agora, dec.distancia), res

        sequencia = 0
        for d in reversed(self._historico):
            if dec.aceito and d.aceito and d.usuario_id == dec.usuario_id:
                sequencia += 1
            else:
                break
        self._situacao = Situacao(Estado.ANALISANDO, "Verificando identidade...",
                                  distancia=dec.distancia, progresso=min(1.0, sequencia / n_ok))
        return self._situacao, res

    # ---- decisões --------------------------------------------------------------------
    def _autorizar(self, usuario_id: int, distancia: float, agora: float) -> Situacao:
        usuario = self.banco.obter_usuario(usuario_id)
        ok, motivo = autorizado(usuario, self.zona)
        if not ok:
            return self._negar(f"nao_autorizado:{motivo}", usuario, agora, distancia)
        if self.cfg.pin_obrigatorio.get(self.zona.nivel, False):
            if not usuario.tem_pin:
                return self._negar("pin_nao_cadastrado", usuario, agora, distancia)
            self._usuario_pin, self._dist_pin = usuario, distancia
            self._pin_tentativas = 0
            self._pin_ate = agora + self.cfg.pin_prazo_s
            self._situacao = Situacao(Estado.PIN_NECESSARIO, f"{usuario.nome}: informe o PIN",
                                      usuario, distancia, 1.0)
            return self._situacao
        return self._conceder(usuario, distancia, agora)

    def cancelar_pin(self) -> Situacao:
        """O usuário desistiu de informar o PIN: conta como tentativa negada."""
        if self._situacao.estado != Estado.PIN_NECESSARIO or self._usuario_pin is None:
            return self._situacao
        return self._negar("pin_cancelado", self._usuario_pin, self._relogio(), self._dist_pin)

    def informar_pin(self, pin: str) -> Situacao:
        if self._situacao.estado != Estado.PIN_NECESSARIO or self._usuario_pin is None:
            return self._situacao
        agora = self._relogio()
        if agora > self._pin_ate:
            return self._negar("pin_expirado", self._usuario_pin, agora, self._dist_pin)
        if self.banco.verificar_pin(self._usuario_pin.id, pin):
            return self._conceder(self._usuario_pin, self._dist_pin, agora)
        self._pin_tentativas += 1
        if self._pin_tentativas >= self.cfg.pin_tentativas:
            return self._negar("pin_incorreto", self._usuario_pin, agora, self._dist_pin)
        self._situacao = Situacao(
            Estado.PIN_NECESSARIO, f"PIN incorreto ({self._pin_tentativas}/{self.cfg.pin_tentativas})",
            self._usuario_pin, self._dist_pin, 1.0)
        return self._situacao

    def _conceder(self, usuario: Usuario, distancia: Optional[float], agora: float) -> Situacao:
        self.banco.registrar_acesso("CONCEDIDO", self.zona.id, usuario.id, usuario.nome, "ok", distancia)
        self._falhas = 0
        self._exibir_ate = agora + self.cfg.resultado_exibido_s
        self._situacao = Situacao(Estado.CONCEDIDO,
                                  f"Acesso concedido: {usuario.nome} - {NOMES_NIVEL[usuario.nivel]}",
                                  usuario, distancia, 1.0)
        return self._situacao

    def _negar(self, motivo: str, usuario: Optional[Usuario], agora: float,
               distancia: Optional[float]) -> Situacao:
        self.banco.registrar_acesso("NEGADO", self.zona.id, usuario.id if usuario else None,
                                    usuario.nome if usuario else None, motivo, distancia)
        self._falhas += 1
        self._exibir_ate = agora + self.cfg.resultado_exibido_s
        texto = MENSAGENS.get(motivo, motivo)
        self._situacao = Situacao(Estado.NEGADO, f"Acesso negado: {texto}", usuario, distancia, 0.0)
        if self._falhas >= self.cfg.max_falhas:
            self._bloqueado_ate = agora + self.cfg.bloqueio_s
            self.banco.registrar_acesso("BLOQUEIO", self.zona.id, None, None,
                                        f"{self._falhas}_negativas_seguidas", None)
            self._falhas = 0
        return self._situacao
