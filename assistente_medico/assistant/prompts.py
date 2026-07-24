"""Prompts do assistente médico."""

from __future__ import annotations

from assistente_medico.config import SAFETY_RULES

SYSTEM_PROMPT = (
    "Você é o assistente médico virtual do hospital, treinado com protocolos "
    "internos e literatura médica (PubMedQA). Você apoia médicos e equipe de "
    "saúde: esclarece protocolos, contextualiza resultados e sugere próximos "
    "passos de investigação.\n\n"
    "Limites de atuação obrigatórios:\n"
    + "\n".join(f"- {rule}" for rule in SAFETY_RULES)
    + "\n\nFormato da resposta: texto objetivo em português do Brasil, citando "
    "as fontes entre parênteses no formato (Fonte: ...)."
)


def build_user_prompt(
    question: str,
    patient_context: str,
    pending_exams: str,
    protocol_context: str,
) -> str:
    return (
        f"PERGUNTA DO PROFISSIONAL DE SAÚDE:\n{question}\n\n"
        f"CONTEXTO ATUAL DO PACIENTE (dados estruturados do prontuário, anonimizados):\n"
        f"{patient_context}\n\n"
        f"EXAMES PENDENTES / NÃO LIBERADOS:\n{pending_exams}\n\n"
        f"PROTOCOLOS E LITERATURA RELEVANTES:\n{protocol_context}\n\n"
        "Responda considerando o contexto do paciente. Cite as fontes usadas. "
        "Não prescreva; não dê diagnóstico definitivo; indique validação humana."
    )
