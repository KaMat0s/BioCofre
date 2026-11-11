"""Cadastro de usuários: avaliação de qualidade das amostras e gravação dos modelos."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .banco import Banco, Usuario
from .config import Config
from .pipeline.classificacao import Galeria
from .pipeline.executor import Pipeline, ResultadoPipeline
from .pipeline.preprocessamento import Qualidade, medir_qualidade

MENSAGENS_AMOSTRA = {
    "ok": "Amostra capturada",
    "sem_rosto": "Nenhum rosto detectado",
    "varios_rostos": "Apenas uma pessoa na imagem",
    "imagem_desfocada": "Imagem desfocada: fique parado",
    "imagem_escura": "Pouca luz: melhore a iluminação",
    "imagem_estourada": "Luz excessiva no rosto",
    "baixo_contraste": "Baixo contraste: ajuste a iluminação",
}


class ErroCadastro(ValueError):
    """Dados de cadastro inválidos."""


@dataclass
class Amostra:
    aceita: bool
    motivo: str
    vetor: Optional[np.ndarray] = None
    qualidade: Optional[Qualidade] = None
    resultado: Optional[ResultadoPipeline] = None


class Cadastro:
    def __init__(self, banco: Banco, pipeline: Pipeline, cfg: Config) -> None:
        self.banco, self.pipeline, self.cfg = banco, pipeline, cfg

    def avaliar_quadro(self, quadro: np.ndarray, ja_recortado: bool = False,
                       verificar_qualidade: bool = True) -> Amostra:
        """Roda as fases 2 a 4 e confere se o quadro serve como amostra de cadastro."""
        res = self.pipeline.extrair(quadro, ja_recortado)
        if res.problema:
            return Amostra(False, res.problema, resultado=res)
        qualidade = medir_qualidade(res.seg.recorte_cinza)
        if verificar_qualidade:
            ok, motivo = qualidade.avaliar(self.cfg)
            if not ok:
                return Amostra(False, motivo, None, qualidade, res)
        return Amostra(True, "ok", res.vetor, qualidade, res)

    def verificar_duplicidade(self, vetores, galeria: Galeria) -> Optional[tuple[Usuario, float]]:
        """Procura um usuário já cadastrado cujo rosto seja parecido com o das novas amostras."""
        if len(galeria) == 0:
            return None
        votos: dict[int, list[float]] = {}
        for v in vetores:
            ranking = galeria.ranking(v)
            if ranking and ranking[0].distancia <= self.cfg.limiar_do_nivel(1):
                votos.setdefault(ranking[0].usuario_id, []).append(ranking[0].distancia)
        for uid, distancias in votos.items():
            if len(distancias) >= max(1, len(vetores) // 2):
                return self.banco.obter_usuario(uid), float(np.mean(distancias))
        return None

    def cadastrar(self, nome: str, cargo: str, nivel: int, divisao: Optional[str], pin: Optional[str],
                  amostras: list[Amostra], galeria: Optional[Galeria] = None,
                  permitir_duplicado: bool = False) -> Usuario:
        nome = nome.strip()
        if not nome:
            raise ErroCadastro("Informe o nome do usuário.")
        if nivel not in (1, 2, 3):
            raise ErroCadastro("O nível deve ser 1, 2 ou 3.")
        divisao = (divisao or "").strip() or None
        if nivel == 2 and not divisao:
            raise ErroCadastro("Diretores (Nível 2) precisam de uma divisão.")
        if pin:
            if not (pin.isdigit() and 4 <= len(pin) <= 8):
                raise ErroCadastro("O PIN deve ter de 4 a 8 dígitos.")
        elif self.cfg.pin_obrigatorio.get(nivel, False):
            raise ErroCadastro(f"O Nível {nivel} exige um PIN de 4 a 8 dígitos.")
        validas = [a for a in amostras if a.aceita and a.vetor is not None]
        if len(validas) < 5:
            raise ErroCadastro("São necessárias pelo menos 5 amostras de boa qualidade.")
        if galeria is not None and not permitir_duplicado:
            parecido = self.verificar_duplicidade([a.vetor for a in validas], galeria)
            if parecido:
                usuario, d = parecido
                raise ErroCadastro(f"O rosto é muito parecido com o de '{usuario.nome}' "
                                   f"(distância {d:.3f}). Cadastro recusado.")
        usuario = self.banco.criar_usuario(nome, cargo.strip(), nivel, divisao, pin,
                                           [a.vetor for a in validas],
                                           [a.qualidade.nitidez for a in validas])
        self.banco.registrar_acesso("CADASTRO", None, usuario.id, usuario.nome,
                                    f"nivel_{nivel};{len(validas)}_amostras")
        return usuario
