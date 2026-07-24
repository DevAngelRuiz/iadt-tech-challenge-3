"""Logging detalhado para rastreamento e auditoria (R7).

Cada interação com o assistente gera um registro JSONL com: timestamp,
sessão, paciente, pergunta, etapas executadas no grafo, tools chamadas,
resposta bruta da LLM, violações de guardrail, fontes e resposta final.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from assistente_medico.config import AUDIT_LOG_PATH

logger = logging.getLogger(__name__)


class AuditLogger:
    def __init__(self, log_path: Path | str | None = None):
        self.log_path = Path(log_path) if log_path else AUDIT_LOG_PATH
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def new_session_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def log_interaction(self, record: dict[str, Any]) -> dict[str, Any]:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **record,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        logger.info(
            "Auditoria registrada: sessao=%s paciente=%s etapas=%d violacoes=%s",
            entry.get("session_id"),
            entry.get("paciente_id"),
            len(entry.get("etapas", [])),
            entry.get("guardrail_violations") or "nenhuma",
        )
        return entry

    def read_all(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        entries = []
        with self.log_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
