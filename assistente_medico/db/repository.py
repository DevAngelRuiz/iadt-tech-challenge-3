"""Consultas estruturadas a prontuários e registros (R4, R5).

Camada de acesso usada pelas tools do LangChain: só expõe dados clínicos
não identificáveis (a base é sintética, mas o desenho segue anonimização).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from assistente_medico.config import DB_PATH
from assistente_medico.db.schema import SCHEMA_SQL


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class HospitalRepository:
    """Repositório SQLite de pacientes, exames, prescrições e alertas."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ── Pacientes ──────────────────────────────────────────────────────

    def upsert_paciente(self, paciente: dict[str, Any]) -> None:
        self._conn.execute(
            """INSERT INTO pacientes
               (id, sexo, idade, alergias, comorbidades, medicamentos_em_uso,
                hipotese_diagnostica, setor, atualizado_em)
               VALUES (:id, :sexo, :idade, :alergias, :comorbidades,
                       :medicamentos_em_uso, :hipotese_diagnostica, :setor, :atualizado_em)
               ON CONFLICT(id) DO UPDATE SET
                 sexo=excluded.sexo, idade=excluded.idade, alergias=excluded.alergias,
                 comorbidades=excluded.comorbidades,
                 medicamentos_em_uso=excluded.medicamentos_em_uso,
                 hipotese_diagnostica=excluded.hipotese_diagnostica,
                 setor=excluded.setor, atualizado_em=excluded.atualizado_em""",
            {**paciente, "atualizado_em": _now()},
        )
        self._conn.commit()

    def get_paciente(self, paciente_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM pacientes WHERE id = ?", (paciente_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_pacientes(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM pacientes ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    # ── Exames ─────────────────────────────────────────────────────────

    def add_exame(
        self,
        paciente_id: str,
        nome: str,
        status: str = "pendente",
        resultado: str | None = None,
        critico: bool = False,
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO exames (paciente_id, nome, status, resultado, critico,
                                   solicitado_em, liberado_em)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                paciente_id,
                nome,
                status,
                resultado,
                int(critico),
                _now(),
                _now() if status == "liberado" else None,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def exames_do_paciente(self, paciente_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM exames WHERE paciente_id = ? ORDER BY id", (paciente_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def exames_pendentes(self, paciente_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM exames WHERE paciente_id = ? AND status != 'liberado' ORDER BY id",
            (paciente_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def resultados_criticos(self, paciente_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM exames WHERE paciente_id = ? AND critico = 1 ORDER BY id",
            (paciente_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Prescrições pendentes de validação humana ──────────────────────

    def add_prescricao_pendente(self, paciente_id: str, descricao: str) -> int:
        cur = self._conn.execute(
            """INSERT INTO prescricoes_pendentes (paciente_id, descricao, criado_em)
               VALUES (?, ?, ?)""",
            (paciente_id, descricao, _now()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def prescricoes_pendentes(self, paciente_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT * FROM prescricoes_pendentes
               WHERE paciente_id = ? AND status = 'aguardando_validacao' ORDER BY id""",
            (paciente_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Alertas para a equipe médica ───────────────────────────────────

    def registrar_alerta(self, paciente_id: str, tipo: str, mensagem: str) -> int:
        cur = self._conn.execute(
            """INSERT INTO alertas (paciente_id, tipo, mensagem, criado_em)
               VALUES (?, ?, ?, ?)""",
            (paciente_id, tipo, mensagem, _now()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def alertas_do_paciente(self, paciente_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM alertas WHERE paciente_id = ? ORDER BY id", (paciente_id,)
        ).fetchall()
        return [dict(r) for r in rows]
