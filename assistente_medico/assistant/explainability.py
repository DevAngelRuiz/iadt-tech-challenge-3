"""Explainability (R8): rastreio das fontes usadas em cada resposta.

Cada resposta do assistente carrega uma lista de fontes no formato:
- ``protocolo:PROTO-002`` — protocolo interno do hospital;
- ``pubmedqa:12345678``  — literatura médica (pubid do PubMedQA);
- ``prontuario:PAC-001/exames`` — dado estruturado do paciente;
- ``politica_seguranca`` — regra institucional de limite de atuação.
"""

from __future__ import annotations

import re
from typing import Any

_SOURCE_IN_TEXT = re.compile(r"\(Fonte: ([^)]+)\)")


def extract_sources_from_text(text: str) -> list[str]:
    """Extrai citações '(Fonte: ...)' produzidas pela LLM."""
    sources: list[str] = []
    for match in _SOURCE_IN_TEXT.findall(text):
        for part in re.split(r"[;,]", match):
            cleaned = part.strip()
            if cleaned and cleaned not in sources:
                sources.append(cleaned)
    return sources


def build_source_list(
    context_docs: list[dict[str, Any]],
    patient_id: str | None,
    used_patient_fields: list[str],
    response_text: str,
) -> list[str]:
    """Consolida fontes: documentos recuperados + prontuário + citações da LLM."""
    sources: list[str] = []
    for doc in context_docs:
        ref = doc.get("ref")
        if ref and ref not in sources:
            sources.append(ref)
    if patient_id:
        for f in used_patient_fields:
            ref = f"prontuario:{patient_id}/{f}"
            if ref not in sources:
                sources.append(ref)
    for cited in extract_sources_from_text(response_text):
        if cited not in sources:
            sources.append(cited)
    return sources
