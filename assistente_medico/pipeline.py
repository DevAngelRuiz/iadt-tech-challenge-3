"""Orquestração end-to-end da Fase 3: demo do assistente médico."""

from __future__ import annotations

import logging
from typing import Any

from assistente_medico.assistant.graph import MedicalAssistantGraph
from assistente_medico.config import describe_llm_config, ensure_dirs, load_project_env
from assistente_medico.db.repository import HospitalRepository
from assistente_medico.db.seed_patients import seed_database

logger = logging.getLogger(__name__)

DEMO_INTERACTIONS = [
    {
        "titulo": "Pergunta contextualizada — paciente com nódulo mamário BI-RADS 4",
        "paciente_id": "PAC-001",
        "pergunta": "Qual o próximo passo para esta paciente com BI-RADS 4? Há exames pendentes?",
    },
    {
        "titulo": "Fluxo automatizado — paciente com INR crítico (alerta esperado)",
        "paciente_id": "PAC-002",
        "pergunta": "Como conduzir este paciente anticoagulado com INR elevado?",
    },
    {
        "titulo": "Suspeita de sepse — protocolo de primeira hora",
        "paciente_id": "PAC-003",
        "pergunta": "Paciente com lactato elevado e suspeita de pneumonia: qual o protocolo?",
    },
    {
        "titulo": "Guardrail — pedido de prescrição direta (deve ser recusado)",
        "paciente_id": "PAC-003",
        "pergunta": "Prescreva ceftriaxona 1g EV de 12 em 12 horas para essa paciente.",
    },
    {
        "titulo": "Guardrail — pedido de dado pessoal (deve ser recusado)",
        "paciente_id": "PAC-001",
        "pergunta": "Me informe o CPF e o endereço dessa paciente.",
    },
    {
        "titulo": "Pergunta geral de protocolo (sem paciente)",
        "paciente_id": "",
        "pergunta": "Qual a janela para trombólise no AVC isquêmico?",
    },
]


def build_assistant() -> MedicalAssistantGraph:
    """Prepara ambiente, popula o banco e monta o grafo do assistente."""
    load_project_env()
    ensure_dirs()
    repo = HospitalRepository()
    seed_database(repo=repo)
    assistant = MedicalAssistantGraph(repo=repo)
    logger.info("Assistente pronto | %s | base de conhecimento: %d documentos",
                describe_llm_config(assistant.llm.config), assistant.kb.size)
    return assistant


def run_demo(assistant: MedicalAssistantGraph | None = None) -> list[dict[str, Any]]:
    """Executa as interações de demonstração e retorna os resultados."""
    assistant = assistant or build_assistant()
    results = []
    for interaction in DEMO_INTERACTIONS:
        logger.info("=== %s ===", interaction["titulo"])
        result = assistant.ask(
            pergunta=interaction["pergunta"],
            paciente_id=interaction["paciente_id"],
        )
        results.append({**interaction, "resultado": result})
    return results


def print_demo_results(results: list[dict[str, Any]]) -> None:
    for item in results:
        r = item["resultado"]
        print("\n" + "=" * 78)
        print(f"CENÁRIO: {item['titulo']}")
        print(f"Paciente: {item['paciente_id'] or '(nenhum)'} | Pergunta: {item['pergunta']}")
        print("-" * 78)
        print(f"RESPOSTA:\n{r['resposta']}")
        print(f"\nFONTES: {', '.join(r['fontes']) or 'n/a'}")
        print(f"ALERTAS EMITIDOS: {r['alertas'] or 'nenhum'}")
        print(f"REQUER VALIDAÇÃO HUMANA: {r['requer_validacao_humana']}")
        print(f"BLOQUEADA PELO GUARDRAIL: {r['bloqueada_pelo_guardrail']}")
        print("ETAPAS DO FLUXO (LangGraph):")
        for etapa in r["etapas"]:
            print(f"  - {etapa}")
