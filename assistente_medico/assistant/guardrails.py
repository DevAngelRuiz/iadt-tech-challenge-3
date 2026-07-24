"""Guardrails de segurança do assistente (R6).

Limites de atuação:
- Bloquear pedidos de prescrição direta, diagnóstico definitivo e PII;
- Sanear respostas da LLM que contenham posologia/prescrição;
- Forçar disclaimer e flag `requer_validacao_humana` em toda resposta.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from assistente_medico.config import DISCLAIMER

# Pedidos que o assistente deve recusar explicitamente.
_BLOCKED_REQUEST_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "pedido_de_prescricao",
        re.compile(
            r"\b(prescrev[aeo]\w*|receit[ea]\s|me\s+d[êe]\s+a\s+dose|"
            r"qual\s+dose\s+(?:eu\s+)?(?:dou|administro)|aumente\s+a\s+dose|"
            r"ajuste\s+a\s+dose|libera\s+a\s+medica[çc][ãa]o)",
            re.IGNORECASE,
        ),
    ),
    (
        "pedido_de_diagnostico_definitivo",
        re.compile(
            r"\b(diagn[óo]stico\s+definitivo|feche\s+o\s+diagn[óo]stico|"
            r"confirme\s+o\s+diagn[óo]stico|qual\s+[ée]\s+o\s+diagn[óo]stico\s+final)",
            re.IGNORECASE,
        ),
    ),
    (
        "pedido_de_dados_pessoais",
        re.compile(
            r"\b(cpf|rg\b|endere[çc]o|telefone|nome\s+completo|data\s+de\s+nascimento)\b",
            re.IGNORECASE,
        ),
    ),
]

# Conteúdo proibido na SAÍDA da LLM (posologia explícita = prescrição).
_FORBIDDEN_OUTPUT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "posologia_explicita",
        re.compile(
            r"\b(tome|administre|prescrevo|aplique)\b[^.\n]{0,80}?"
            r"\d+\s*(mg|g|ml|mcg|ui|gotas|comprimidos?)",
            re.IGNORECASE,
        ),
    ),
    (
        "diagnostico_afirmativo",
        re.compile(
            r"\bo\s+diagn[óo]stico\s+(?:definitivo\s+)?[ée]\b",
            re.IGNORECASE,
        ),
    ),
]

REFUSAL_MESSAGES = {
    "pedido_de_prescricao": (
        "Não posso prescrever medicamentos, definir posologia ou ajustar doses: "
        "prescrição é ato médico e exige validação humana. Posso indicar o protocolo "
        "institucional aplicável e os pontos de checagem (alergias, função renal, "
        "interações) para apoiar a decisão do prescritor."
    ),
    "pedido_de_diagnostico_definitivo": (
        "Não forneço diagnóstico definitivo: isso exige avaliação clínica humana. "
        "Posso apoiar com os critérios dos protocolos institucionais e com a "
        "literatura, citando as fontes."
    ),
    "pedido_de_dados_pessoais": (
        "Não posso fornecer dados pessoais identificáveis de pacientes. O acesso é "
        "restrito e auditado conforme a política de privacidade do hospital."
    ),
}


@dataclass
class GuardrailResult:
    allowed: bool
    final_text: str
    violations: list[str] = field(default_factory=list)
    sanitized: bool = False
    requer_validacao_humana: bool = True


def check_request(question: str) -> GuardrailResult:
    """Avalia a PERGUNTA antes de chamar a LLM. Bloqueia pedidos proibidos."""
    for label, pattern in _BLOCKED_REQUEST_PATTERNS:
        if pattern.search(question):
            refusal = f"{REFUSAL_MESSAGES[label]}\n\n{DISCLAIMER}"
            return GuardrailResult(
                allowed=False,
                final_text=refusal,
                violations=[label],
            )
    return GuardrailResult(allowed=True, final_text="")


def check_response(text: str) -> GuardrailResult:
    """Avalia a RESPOSTA da LLM. Sanitiza trechos proibidos e anexa disclaimer."""
    violations: list[str] = []
    sanitized_text = text
    for label, pattern in _FORBIDDEN_OUTPUT_PATTERNS:
        if pattern.search(sanitized_text):
            violations.append(label)
            sanitized_text = pattern.sub("[TRECHO REMOVIDO PELO GUARDRAIL DE SEGURANÇA]", sanitized_text)

    if DISCLAIMER not in sanitized_text:
        sanitized_text = f"{sanitized_text}\n\n{DISCLAIMER}"

    return GuardrailResult(
        allowed=True,
        final_text=sanitized_text,
        violations=violations,
        sanitized=bool(violations),
    )
