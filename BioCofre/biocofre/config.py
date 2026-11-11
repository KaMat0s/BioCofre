"""Parâmetros do sistema, reunidos em uma única estrutura serializável em JSON.

Todos os módulos recebem uma instância de ``Config``; nenhum valor numérico de
decisão fica espalhado pelo código. Os limiares de distância são por nível de
segurança da zona e devem ser recalibrados com imagens do próprio ambiente
(comando ``calibrar`` da linha de comando).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


def _por_nivel(a, b, c) -> dict[int, float]:
    return {1: a, 2: b, 3: c}


@dataclass
class Config:
    # --- Fase 1: aquisição ---------------------------------------------------
    camera_indice: int = 0
    largura: int = 640
    altura: int = 480
    espelhar: bool = True

    # --- Fase 2: pré-processamento -------------------------------------------
    gauss_kernel: int = 3          # tamanho do filtro gaussiano (0 ou 1 desliga)
    clahe_clip: float = 2.0        # limite de contraste do CLAHE (reserva da detecção; 0 desliga)
    clahe_grade: int = 8           # blocos do CLAHE por lado
    equalizar_caracteristicas: bool = False  # usa a imagem equalizada também no descritor LBP

    # --- Fase 3: segmentação -------------------------------------------------
    haar_escala: float = 1.1
    haar_vizinhos: int = 5
    face_min_px: int = 80          # menor rosto aceito, em pixels do quadro
    tam_face: int = 100            # lado do recorte normalizado
    margem_face: float = 0.20      # folga ao redor da caixa do detector
    mascara_eliptica: bool = True
    mascara_rx: float = 0.40       # semieixo horizontal da elipse (fração do lado do recorte)
    mascara_ry: float = 0.50       # semieixo vertical da elipse

    # --- Fase 4: características (LBP uniforme em grade) ----------------------
    lbp_grade: int = 8

    # --- Fase 5: classificação e política de acesso ----------------------------
    limiar: dict = field(default_factory=lambda: _por_nivel(0.422, 0.396, 0.386))
    razao_max: float = 0.90                  # d1/d2 máximo; 1.0 desliga a margem
    quadros_confirmacao: dict = field(default_factory=lambda: {1: 3, 2: 5, 3: 8})
    pin_obrigatorio: dict = field(default_factory=lambda: {1: False, 2: False, 3: True})
    quadros_negacao: int = 15
    tempo_max_analise_s: float = 10.0
    max_falhas: int = 3
    bloqueio_s: int = 60
    resultado_exibido_s: float = 4.0
    pin_tentativas: int = 3
    pin_prazo_s: float = 30.0
    sem_rosto_reinicia_s: float = 2.0   # sem rosto por este tempo, a análise recomeça
    rearme_sem_rosto_s: float = 1.5     # após uma decisão, o rosto precisa sair do quadro

    # --- Cadastro ---------------------------------------------------------------
    amostras_cadastro: int = 15
    nitidez_min: float = 40.0
    brilho_min: float = 50.0
    brilho_max: float = 210.0
    contraste_min: float = 25.0

    # --- Persistência -----------------------------------------------------------
    pasta_dados: str = "dados"

    # ------------------------------------------------------------------------
    @property
    def caminho_banco(self) -> Path:
        return Path(self.pasta_dados) / "biocofre.db"

    @property
    def caminho_chave(self) -> Path:
        return Path(self.pasta_dados) / "chave_log.key"

    def limiar_do_nivel(self, nivel: int) -> float:
        return float(self.limiar[int(nivel)])

    def salvar(self, caminho: str | Path) -> None:
        Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        Path(caminho).write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def carregar(cls, caminho: str | Path | None = None) -> "Config":
        """Carrega um JSON (chaves de dicionário viram inteiros). Sem arquivo, usa o padrão."""
        cfg = cls()
        if caminho is None or not Path(caminho).exists():
            return cfg
        bruto = json.loads(Path(caminho).read_text(encoding="utf-8"))
        validos = {f.name for f in fields(cls)}
        for chave, valor in bruto.items():
            if chave not in validos:
                continue
            if isinstance(valor, dict):
                valor = {int(k): v for k, v in valor.items()}
            setattr(cfg, chave, valor)
        return cfg
