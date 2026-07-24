"""Anonimização de dados médicos (requisito R2).

Remove/substitui PII antes de qualquer texto entrar no dataset de fine-tuning
ou nos prompts do assistente: CPF, RG, telefone, e-mail, datas de nascimento,
nomes precedidos de pronomes de tratamento e números de prontuário reais.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Padrões de PII em texto PT-BR (ordem importa: mais específicos primeiro).
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("CPF", re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")),
    ("RG", re.compile(r"\bRG\s*:?\s*[\d.\-xX]{5,12}\b", re.IGNORECASE)),
    ("TELEFONE", re.compile(r"\b(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?9?\d{4}[-\s]?\d{4}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),
    ("DATA_NASCIMENTO", re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")),
    ("PRONTUARIO", re.compile(r"\bprontu[áa]rio\s*(?:n[ºo°.]?\s*)?\d{3,}\b", re.IGNORECASE)),
    ("CEP", re.compile(r"\b\d{5}-?\d{3}\b")),
    (
        "NOME",
        re.compile(
            r"\b(?:Sr\.?|Sra\.?|Dr\.?|Dra\.?|Paciente|paciente)\s+"
            r"([A-ZÀ-Ú][a-zà-ú]+(?:\s+(?:d[aeo]s?\s+)?[A-ZÀ-Ú][a-zà-ú]+)+)"
        ),
    ),
]


@dataclass
class AnonymizationResult:
    text: str
    replacements: dict[str, int] = field(default_factory=dict)

    @property
    def total_replacements(self) -> int:
        return sum(self.replacements.values())


def anonymize_text(text: str) -> AnonymizationResult:
    """Substitui PII por placeholders do tipo [CPF_REMOVIDO]."""
    replacements: dict[str, int] = {}
    result = text
    for label, pattern in _PII_PATTERNS:
        if label == "NOME":
            # Preserva o pronome de tratamento, remove apenas o nome próprio.
            def _sub_nome(match: re.Match[str]) -> str:
                prefix = match.group(0)[: match.start(1) - match.start(0)]
                return f"{prefix}[NOME_REMOVIDO]"

            result, count = pattern.subn(_sub_nome, result)
        else:
            result, count = pattern.subn(f"[{label}_REMOVIDO]", result)
        if count:
            replacements[label] = replacements.get(label, 0) + count
    return AnonymizationResult(text=result, replacements=replacements)


def contains_pii(text: str) -> bool:
    return any(pattern.search(text) for _, pattern in _PII_PATTERNS)


def anonymize_record(record: dict, text_fields: tuple[str, ...]) -> tuple[dict, int]:
    """Anonimiza campos textuais de um registro; retorna (registro, nº de remoções)."""
    out = dict(record)
    total = 0
    for f in text_fields:
        value = out.get(f)
        if isinstance(value, str):
            res = anonymize_text(value)
            out[f] = res.text
            total += res.total_replacements
    return out, total
