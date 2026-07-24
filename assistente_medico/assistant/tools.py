"""Tools LangChain do assistente (R4): consultas estruturadas e alertas.

As tools encapsulam o `HospitalRepository` (SQLite) e a `KnowledgeBase`,
permitindo que o fluxo LangChain/LangGraph consulte prontuários, exames,
protocolos e registre alertas para a equipe médica.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from assistente_medico.assistant.retrieval import KnowledgeBase
from assistente_medico.db.repository import HospitalRepository


def build_tools(repo: HospitalRepository, kb: KnowledgeBase) -> list[Any]:
    """Cria as tools LangChain ligadas ao repositório e à base de conhecimento."""

    @tool
    def buscar_paciente(paciente_id: str) -> str:
        """Busca os dados clínicos estruturados (anonimizados) de um paciente pelo ID sintético (ex.: PAC-001)."""
        paciente = repo.get_paciente(paciente_id)
        if not paciente:
            return json.dumps({"erro": f"Paciente {paciente_id} não encontrado."}, ensure_ascii=False)
        return json.dumps(paciente, ensure_ascii=False)

    @tool
    def listar_exames_pendentes(paciente_id: str) -> str:
        """Lista os exames pendentes ou não liberados de um paciente."""
        exames = repo.exames_pendentes(paciente_id)
        return json.dumps(exames, ensure_ascii=False)

    @tool
    def listar_resultados_criticos(paciente_id: str) -> str:
        """Lista resultados de exames marcados como críticos para um paciente."""
        criticos = repo.resultados_criticos(paciente_id)
        return json.dumps(criticos, ensure_ascii=False)

    @tool
    def buscar_protocolo(consulta: str) -> str:
        """Busca protocolos internos e literatura médica relevantes para uma consulta clínica."""
        docs = kb.search(consulta, top_k=3)
        return json.dumps(docs, ensure_ascii=False)

    @tool
    def registrar_alerta(paciente_id: str, tipo: str, mensagem: str) -> str:
        """Registra um alerta para a equipe médica (ex.: resultado crítico, exame pendente bloqueando conduta)."""
        alerta_id = repo.registrar_alerta(paciente_id, tipo, mensagem)
        return json.dumps({"alerta_id": alerta_id, "status": "registrado"}, ensure_ascii=False)

    return [
        buscar_paciente,
        listar_exames_pendentes,
        listar_resultados_criticos,
        buscar_protocolo,
        registrar_alerta,
    ]
