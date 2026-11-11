"""Persistência em SQLite: usuários, modelos biométricos, zonas e log de auditoria.

O log é encadeado: cada registro guarda o HMAC-SHA256 do seu conteúdo somado ao HMAC do
registro anterior, com uma chave secreta guardada fora do banco. Alterar ou apagar um
registro do meio da cadeia, sem a chave, é detectado por ``verificar_integridade``.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

GENESE = "0" * 64
ZONAS_PADRAO = [
    ("Arquivo geral", 1, None),
    ("Acervo da Divisão de Toxicologia", 2, "Toxicologia"),
    ("Acervo da Divisão de Biossegurança", 2, "Biossegurança"),
    ("Cofre de toxinas", 3, None),
]


class ErroBanco(RuntimeError):
    """Violação de regra de negócio ou falha de persistência."""


@dataclass
class Usuario:
    id: int
    nome: str
    cargo: str
    nivel: int
    divisao: Optional[str]
    ativo: bool
    tem_pin: bool


@dataclass
class Zona:
    id: int
    nome: str
    nivel: int
    divisao: Optional[str]


@dataclass
class Integridade:
    ok: bool
    total: int
    primeiro_invalido: Optional[int]
    mensagem: str


_ESQUEMA = """
CREATE TABLE IF NOT EXISTS usuarios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nome TEXT NOT NULL UNIQUE,
    cargo TEXT NOT NULL DEFAULT '',
    nivel INTEGER NOT NULL CHECK (nivel BETWEEN 1 AND 3),
    divisao TEXT,
    pin_sal BLOB,
    pin_hash BLOB,
    ativo INTEGER NOT NULL DEFAULT 1,
    criado_em TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS modelos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    usuario_id INTEGER NOT NULL REFERENCES usuarios(id) ON DELETE CASCADE,
    vetor BLOB NOT NULL,
    qualidade REAL,
    criado_em TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS zonas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nome TEXT NOT NULL UNIQUE,
    nivel INTEGER NOT NULL CHECK (nivel BETWEEN 1 AND 3),
    divisao TEXT
);
CREATE TABLE IF NOT EXISTS acessos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    zona_id INTEGER REFERENCES zonas(id),
    usuario_id INTEGER,
    usuario_nome TEXT,
    evento TEXT NOT NULL,
    motivo TEXT,
    distancia REAL,
    hash_ant TEXT NOT NULL,
    hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS parametros (
    chave TEXT PRIMARY KEY,
    valor BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_acessos_ts ON acessos(ts);
CREATE INDEX IF NOT EXISTS ix_modelos_usuario ON modelos(usuario_id);
"""


def _agora() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Banco:
    def __init__(self, caminho: str | Path, caminho_chave: str | Path,
                 iteracoes_kdf: int = 200_000) -> None:
        Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(caminho))
        self._con.row_factory = sqlite3.Row
        self._con.execute("PRAGMA foreign_keys = ON")
        self._con.executescript(_ESQUEMA)
        self._iteracoes = iteracoes_kdf
        self._chave = self._carregar_chave(Path(caminho_chave))
        self._semear_zonas()

    # ---- infraestrutura -------------------------------------------------------------
    @staticmethod
    def _carregar_chave(caminho: Path) -> bytes:
        if caminho.exists():
            return bytes.fromhex(caminho.read_text().strip())
        caminho.parent.mkdir(parents=True, exist_ok=True)
        chave = secrets.token_bytes(32)
        caminho.write_text(chave.hex())
        try:
            os.chmod(caminho, 0o600)
        except OSError:
            pass
        return chave

    def _semear_zonas(self) -> None:
        if self._con.execute("SELECT COUNT(*) FROM zonas").fetchone()[0] == 0:
            self._con.executemany("INSERT INTO zonas (nome, nivel, divisao) VALUES (?, ?, ?)", ZONAS_PADRAO)
            self._con.commit()

    def fechar(self) -> None:
        self._con.close()

    def _kdf(self, segredo: str, sal: bytes) -> bytes:
        return hashlib.pbkdf2_hmac("sha256", segredo.encode("utf-8"), sal, self._iteracoes)

    # ---- usuários -------------------------------------------------------------------
    @staticmethod
    def _usuario(linha: sqlite3.Row) -> Usuario:
        return Usuario(linha["id"], linha["nome"], linha["cargo"], linha["nivel"], linha["divisao"],
                       bool(linha["ativo"]), linha["pin_hash"] is not None)

    def criar_usuario(self, nome: str, cargo: str, nivel: int, divisao: Optional[str],
                      pin: Optional[str], vetores: Iterable[np.ndarray],
                      qualidades: Optional[Iterable[float]] = None) -> Usuario:
        vetores = list(vetores)
        qualidades = list(qualidades) if qualidades is not None else [None] * len(vetores)
        sal = secrets.token_bytes(16) if pin else None
        pin_hash = self._kdf(pin, sal) if pin else None
        try:
            with self._con:
                cur = self._con.execute(
                    "INSERT INTO usuarios (nome, cargo, nivel, divisao, pin_sal, pin_hash, criado_em) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (nome, cargo, nivel, divisao, sal, pin_hash, _agora()))
                uid = cur.lastrowid
                self._con.executemany(
                    "INSERT INTO modelos (usuario_id, vetor, qualidade, criado_em) VALUES (?, ?, ?, ?)",
                    [(uid, np.asarray(v, np.float32).tobytes(), q, _agora())
                     for v, q in zip(vetores, qualidades)])
        except sqlite3.IntegrityError as exc:
            raise ErroBanco(f"Já existe um usuário chamado '{nome}'.") from exc
        return self.obter_usuario(uid)

    def obter_usuario(self, usuario_id: int) -> Optional[Usuario]:
        linha = self._con.execute("SELECT * FROM usuarios WHERE id = ?", (usuario_id,)).fetchone()
        return self._usuario(linha) if linha else None

    def listar_usuarios(self) -> list[Usuario]:
        return [self._usuario(r) for r in self._con.execute("SELECT * FROM usuarios ORDER BY nome")]

    def remover_usuario(self, usuario_id: int) -> None:
        with self._con:
            self._con.execute("DELETE FROM usuarios WHERE id = ?", (usuario_id,))

    def definir_ativo(self, usuario_id: int, ativo: bool) -> None:
        with self._con:
            self._con.execute("UPDATE usuarios SET ativo = ? WHERE id = ?", (int(ativo), usuario_id))

    def verificar_pin(self, usuario_id: int, pin: str) -> bool:
        linha = self._con.execute("SELECT pin_sal, pin_hash FROM usuarios WHERE id = ?",
                                  (usuario_id,)).fetchone()
        if not linha or linha["pin_hash"] is None:
            return False
        return hmac.compare_digest(self._kdf(pin, linha["pin_sal"]), linha["pin_hash"])

    def carregar_modelos(self) -> list[tuple[int, np.ndarray]]:
        """Modelos dos usuários ativos, como (id do usuário, vetor)."""
        consulta = ("SELECT m.usuario_id, m.vetor FROM modelos m JOIN usuarios u ON u.id = m.usuario_id "
                    "WHERE u.ativo = 1 ORDER BY m.id")
        return [(r["usuario_id"], np.frombuffer(r["vetor"], np.float32).copy())
                for r in self._con.execute(consulta)]

    # ---- zonas ----------------------------------------------------------------------
    def zonas(self) -> list[Zona]:
        return [Zona(r["id"], r["nome"], r["nivel"], r["divisao"])
                for r in self._con.execute("SELECT * FROM zonas ORDER BY nivel, nome")]

    def obter_zona(self, zona_id: int) -> Optional[Zona]:
        r = self._con.execute("SELECT * FROM zonas WHERE id = ?", (zona_id,)).fetchone()
        return Zona(r["id"], r["nome"], r["nivel"], r["divisao"]) if r else None

    # ---- senha do administrador -----------------------------------------------------
    def tem_senha_admin(self) -> bool:
        return self._con.execute("SELECT 1 FROM parametros WHERE chave = 'admin_hash'").fetchone() is not None

    def definir_senha_admin(self, senha: str) -> None:
        if len(senha) < 6:
            raise ErroBanco("A senha do administrador deve ter pelo menos 6 caracteres.")
        sal = secrets.token_bytes(16)
        with self._con:
            self._con.execute("INSERT OR REPLACE INTO parametros VALUES ('admin_sal', ?)", (sal,))
            self._con.execute("INSERT OR REPLACE INTO parametros VALUES ('admin_hash', ?)", (self._kdf(senha, sal),))

    def verificar_senha_admin(self, senha: str) -> bool:
        sal = self._con.execute("SELECT valor FROM parametros WHERE chave = 'admin_sal'").fetchone()
        alvo = self._con.execute("SELECT valor FROM parametros WHERE chave = 'admin_hash'").fetchone()
        if not sal or not alvo:
            return False
        return hmac.compare_digest(self._kdf(senha, sal["valor"]), alvo["valor"])

    # ---- log de auditoria -------------------------------------------------------------
    def _assinar(self, campos: list, hash_ant: str) -> str:
        texto = json.dumps(campos + [hash_ant], ensure_ascii=False, separators=(",", ":"))
        return hmac.new(self._chave, texto.encode("utf-8"), hashlib.sha256).hexdigest()

    @staticmethod
    def _campos(ts, zona_id, usuario_id, usuario_nome, evento, motivo, distancia) -> list:
        dist = None if distancia is None else f"{float(distancia):.6f}"
        return [ts, zona_id, usuario_id, usuario_nome, evento, motivo, dist]

    def registrar_acesso(self, evento: str, zona_id: Optional[int], usuario_id: Optional[int] = None,
                         usuario_nome: Optional[str] = None, motivo: str = "",
                         distancia: Optional[float] = None, ts: Optional[str] = None) -> int:
        ts = ts or _agora()
        with self._con:
            ultimo = self._con.execute("SELECT hash FROM acessos ORDER BY id DESC LIMIT 1").fetchone()
            anterior = ultimo["hash"] if ultimo else GENESE
            campos = self._campos(ts, zona_id, usuario_id, usuario_nome, evento, motivo, distancia)
            assinatura = self._assinar(campos, anterior)
            cur = self._con.execute(
                "INSERT INTO acessos (ts, zona_id, usuario_id, usuario_nome, evento, motivo, distancia, "
                "hash_ant, hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ts, zona_id, usuario_id, usuario_nome, evento, motivo,
                 None if distancia is None else float(distancia), anterior, assinatura))
        return cur.lastrowid

    def listar_acessos(self, limite: int = 200, inicio: Optional[str] = None, fim: Optional[str] = None,
                       zona_id: Optional[int] = None, usuario_id: Optional[int] = None,
                       evento: Optional[str] = None) -> list[dict]:
        filtros, params = [], []
        if inicio:
            filtros.append("a.ts >= ?"); params.append(inicio)
        if fim:
            filtros.append("a.ts <= ?"); params.append(fim + "T23:59:59" if len(fim) == 10 else fim)
        if zona_id:
            filtros.append("a.zona_id = ?"); params.append(zona_id)
        if usuario_id:
            filtros.append("a.usuario_id = ?"); params.append(usuario_id)
        if evento:
            filtros.append("a.evento = ?"); params.append(evento)
        onde = ("WHERE " + " AND ".join(filtros)) if filtros else ""
        consulta = (f"SELECT a.id, a.ts, z.nome AS zona, z.nivel AS nivel_zona, a.usuario_nome, a.evento, "
                    f"a.motivo, a.distancia FROM acessos a LEFT JOIN zonas z ON z.id = a.zona_id "
                    f"{onde} ORDER BY a.id DESC" + (" LIMIT ?" if limite else ""))
        if limite:
            params.append(limite)
        return [dict(r) for r in self._con.execute(consulta, params)]

    def verificar_integridade(self) -> Integridade:
        anterior, total = GENESE, 0
        for r in self._con.execute("SELECT * FROM acessos ORDER BY id"):
            total += 1
            campos = self._campos(r["ts"], r["zona_id"], r["usuario_id"], r["usuario_nome"],
                                  r["evento"], r["motivo"], r["distancia"])
            if r["hash_ant"] != anterior or not hmac.compare_digest(self._assinar(campos, anterior), r["hash"]):
                return Integridade(False, total, r["id"],
                                   f"Registro {r['id']} foi alterado ou a cadeia foi quebrada nele.")
            anterior = r["hash"]
        return Integridade(True, total, None, f"Cadeia íntegra: {total} registros verificados.")
