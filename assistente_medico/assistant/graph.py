"""Fluxo de decisão automatizado com LangGraph (R3, R5, R9).

Grafo do assistente médico:

    validar_pergunta ──(bloqueada)──────────────────────────┐
        │ (permitida)                                       │
        ▼                                                   ▼
    carregar_paciente → verificar_exames → recuperar_contexto
        → gerar_resposta → aplicar_guardrails → emitir_alertas → finalizar

Cada nó registra sua etapa para auditoria; o nó final grava o log JSONL.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from assistente_medico.assistant import guardrails
from assistente_medico.assistant.explainability import build_source_list
from assistente_medico.assistant.llm import AssistantLLM
from assistente_medico.assistant.logging_audit import AuditLogger
from assistente_medico.assistant.prompts import SYSTEM_PROMPT, build_user_prompt
from assistente_medico.assistant.retrieval import KnowledgeBase, format_context_docs
from assistente_medico.db.repository import HospitalRepository

logger = logging.getLogger(__name__)


def _append(left: list, right: list) -> list:
    return left + right


class AssistantState(TypedDict, total=False):
    # Entrada
    paciente_id: str
    pergunta: str
    session_id: str
    # Dados coletados pelo fluxo
    paciente: dict[str, Any] | None
    exames_pendentes: list[dict[str, Any]]
    resultados_criticos: list[dict[str, Any]]
    prescricoes_pendentes: list[dict[str, Any]]
    context_docs: list[dict[str, Any]]
    # Geração
    llm_raw: str
    resposta_final: str
    fontes: list[str]
    guardrail_violations: list[str]
    bloqueada: bool
    requer_validacao_humana: bool
    alertas_emitidos: list[dict[str, Any]]
    etapas: Annotated[list[str], _append]


class MedicalAssistantGraph:
    """Monta e executa o grafo LangGraph do assistente médico."""

    def __init__(
        self,
        repo: HospitalRepository | None = None,
        kb: KnowledgeBase | None = None,
        llm: AssistantLLM | None = None,
        audit: AuditLogger | None = None,
    ):
        self.repo = repo or HospitalRepository()
        self.kb = kb or KnowledgeBase()
        self.llm = llm or AssistantLLM()
        self.audit = audit or AuditLogger()
        self.graph = self._build()

    # ── Nós do grafo ───────────────────────────────────────────────────

    def _validar_pergunta(self, state: AssistantState) -> dict[str, Any]:
        result = guardrails.check_request(state["pergunta"])
        if not result.allowed:
            return {
                "bloqueada": True,
                "resposta_final": result.final_text,
                "guardrail_violations": result.violations,
                "fontes": ["politica_seguranca"],
                "requer_validacao_humana": True,
                "etapas": ["validar_pergunta: BLOQUEADA pelos limites de atuação"],
            }
        return {"bloqueada": False, "etapas": ["validar_pergunta: permitida"]}

    def _carregar_paciente(self, state: AssistantState) -> dict[str, Any]:
        paciente = self.repo.get_paciente(state["paciente_id"]) if state.get("paciente_id") else None
        return {
            "paciente": paciente,
            "etapas": [
                f"carregar_paciente: {'encontrado' if paciente else 'não encontrado'}"
            ],
        }

    def _verificar_exames(self, state: AssistantState) -> dict[str, Any]:
        pid = state.get("paciente_id", "")
        pendentes = self.repo.exames_pendentes(pid) if pid else []
        criticos = self.repo.resultados_criticos(pid) if pid else []
        prescricoes = self.repo.prescricoes_pendentes(pid) if pid else []
        return {
            "exames_pendentes": pendentes,
            "resultados_criticos": criticos,
            "prescricoes_pendentes": prescricoes,
            "etapas": [
                f"verificar_exames: {len(pendentes)} pendente(s), "
                f"{len(criticos)} crítico(s), {len(prescricoes)} prescrição(ões) aguardando validação"
            ],
        }

    def _recuperar_contexto(self, state: AssistantState) -> dict[str, Any]:
        query = state["pergunta"]
        paciente = state.get("paciente") or {}
        if paciente.get("hipotese_diagnostica"):
            query = f"{query} {paciente['hipotese_diagnostica']}"
        docs = self.kb.search(query, top_k=3)
        return {
            "context_docs": docs,
            "etapas": [f"recuperar_contexto: {len(docs)} documento(s) — "
                       + ", ".join(d["ref"] for d in docs)],
        }

    def _gerar_resposta(self, state: AssistantState) -> dict[str, Any]:
        paciente = state.get("paciente")
        if paciente:
            patient_context = (
                f"ID: {paciente['id']} | sexo: {paciente['sexo']} | idade: {paciente['idade']}\n"
                f"Alergias: {paciente['alergias'] or 'nenhuma registrada'}\n"
                f"Comorbidades: {paciente['comorbidades'] or 'nenhuma registrada'}\n"
                f"Medicamentos em uso: {paciente['medicamentos_em_uso'] or 'nenhum registrado'}\n"
                f"Hipótese diagnóstica: {paciente['hipotese_diagnostica'] or 'não informada'}\n"
                f"Setor: {paciente['setor']}"
            )
        else:
            patient_context = "Nenhum paciente informado (pergunta geral sobre protocolos)."

        pendentes = state.get("exames_pendentes", [])
        pending_txt = (
            "\n".join(f"- {e['nome']} (status: {e['status']})" for e in pendentes)
            or "Nenhum exame pendente."
        )
        criticos = state.get("resultados_criticos", [])
        if criticos:
            pending_txt += "\nRESULTADOS CRÍTICOS: " + "; ".join(
                f"{e['nome']} = {e['resultado']}" for e in criticos
            )

        user_prompt = build_user_prompt(
            question=state["pergunta"],
            patient_context=patient_context,
            pending_exams=pending_txt,
            protocol_context=format_context_docs(state.get("context_docs", [])),
        )
        start = time.time()
        raw = self.llm.generate(SYSTEM_PROMPT, user_prompt)
        elapsed = time.time() - start
        return {
            "llm_raw": raw,
            "etapas": [f"gerar_resposta: backend={self.llm.backend} em {elapsed:.1f}s"],
        }

    def _aplicar_guardrails(self, state: AssistantState) -> dict[str, Any]:
        result = guardrails.check_response(state.get("llm_raw", ""))
        paciente = state.get("paciente")
        used_fields = []
        if paciente:
            used_fields = ["dados_clinicos"]
            if state.get("exames_pendentes"):
                used_fields.append("exames")
        fontes = build_source_list(
            context_docs=state.get("context_docs", []),
            patient_id=paciente["id"] if paciente else None,
            used_patient_fields=used_fields,
            response_text=result.final_text,
        )
        etapa = "aplicar_guardrails: aprovada"
        if result.sanitized:
            etapa = f"aplicar_guardrails: SANITIZADA ({', '.join(result.violations)})"
        return {
            "resposta_final": result.final_text,
            "guardrail_violations": result.violations,
            "fontes": fontes,
            "requer_validacao_humana": True,
            "etapas": [etapa],
        }

    def _emitir_alertas(self, state: AssistantState) -> dict[str, Any]:
        alertas: list[dict[str, Any]] = []
        pid = state.get("paciente_id", "")
        if pid and state.get("paciente"):
            for exame in state.get("resultados_criticos", []):
                alerta_id = self.repo.registrar_alerta(
                    pid,
                    "resultado_critico",
                    f"Resultado crítico em {exame['nome']}: {exame['resultado']}. "
                    "Comunicar médico responsável em até 30 minutos (PROTO-011).",
                )
                alertas.append({"id": alerta_id, "tipo": "resultado_critico", "exame": exame["nome"]})
            pendentes = state.get("exames_pendentes", [])
            if pendentes:
                alerta_id = self.repo.registrar_alerta(
                    pid,
                    "exames_pendentes",
                    f"{len(pendentes)} exame(s) pendente(s) podem impactar a conduta: "
                    + ", ".join(e["nome"] for e in pendentes),
                )
                alertas.append({"id": alerta_id, "tipo": "exames_pendentes", "qtd": len(pendentes)})
        etapa = (
            f"emitir_alertas: {len(alertas)} alerta(s) registrado(s)"
            if alertas
            else "emitir_alertas: nenhum alerta necessário"
        )
        return {"alertas_emitidos": alertas, "etapas": [etapa]}

    def _finalizar(self, state: AssistantState) -> dict[str, Any]:
        self.audit.log_interaction(
            {
                "session_id": state.get("session_id", ""),
                "paciente_id": state.get("paciente_id", ""),
                "pergunta": state["pergunta"],
                "bloqueada_pelo_guardrail": state.get("bloqueada", False),
                "backend_llm": self.llm.backend,
                "modelo_llm": self.llm.config.get("model"),
                "resposta_bruta_llm": state.get("llm_raw", ""),
                "resposta_final": state.get("resposta_final", ""),
                "guardrail_violations": state.get("guardrail_violations", []),
                "fontes": state.get("fontes", []),
                "alertas_emitidos": state.get("alertas_emitidos", []),
                "requer_validacao_humana": state.get("requer_validacao_humana", True),
                "etapas": state.get("etapas", []),
            }
        )
        return {"etapas": ["finalizar: auditoria gravada"]}

    # ── Montagem ───────────────────────────────────────────────────────

    def _build(self):
        builder = StateGraph(AssistantState)
        builder.add_node("validar_pergunta", self._validar_pergunta)
        builder.add_node("carregar_paciente", self._carregar_paciente)
        builder.add_node("verificar_exames", self._verificar_exames)
        builder.add_node("recuperar_contexto", self._recuperar_contexto)
        builder.add_node("gerar_resposta", self._gerar_resposta)
        builder.add_node("aplicar_guardrails", self._aplicar_guardrails)
        builder.add_node("emitir_alertas", self._emitir_alertas)
        builder.add_node("finalizar", self._finalizar)

        builder.add_edge(START, "validar_pergunta")
        builder.add_conditional_edges(
            "validar_pergunta",
            lambda state: "bloqueada" if state.get("bloqueada") else "permitida",
            {"bloqueada": "finalizar", "permitida": "carregar_paciente"},
        )
        builder.add_edge("carregar_paciente", "verificar_exames")
        builder.add_edge("verificar_exames", "recuperar_contexto")
        builder.add_edge("recuperar_contexto", "gerar_resposta")
        builder.add_edge("gerar_resposta", "aplicar_guardrails")
        builder.add_edge("aplicar_guardrails", "emitir_alertas")
        builder.add_edge("emitir_alertas", "finalizar")
        builder.add_edge("finalizar", END)
        return builder.compile()

    # ── Execução ───────────────────────────────────────────────────────

    def ask(self, pergunta: str, paciente_id: str = "") -> dict[str, Any]:
        initial: AssistantState = {
            "pergunta": pergunta,
            "paciente_id": paciente_id,
            "session_id": self.audit.new_session_id(),
            "etapas": [],
        }
        final_state = self.graph.invoke(initial)
        return {
            "resposta": final_state.get("resposta_final", ""),
            "fontes": final_state.get("fontes", []),
            "alertas": final_state.get("alertas_emitidos", []),
            "exames_pendentes": final_state.get("exames_pendentes", []),
            "requer_validacao_humana": final_state.get("requer_validacao_humana", True),
            "bloqueada_pelo_guardrail": final_state.get("bloqueada", False),
            "etapas": final_state.get("etapas", []),
            "session_id": final_state.get("session_id", ""),
        }
