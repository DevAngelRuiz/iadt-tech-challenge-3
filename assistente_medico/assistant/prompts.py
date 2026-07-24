"""Prompts do assistente médico e classificação de perguntas (meta / fora de escopo)."""

from __future__ import annotations

import re

from assistente_medico.config import DISCLAIMER, SAFETY_RULES

SYSTEM_PROMPT = (
    "Você é o assistente médico virtual do hospital, treinado com protocolos "
    "internos e literatura médica (PubMedQA). Você apoia médicos e equipe de "
    "saúde: esclarece protocolos, contextualiza resultados e sugere próximos "
    "passos de investigação.\n\n"
    "Limites de atuação obrigatórios:\n"
    + "\n".join(f"- {rule}" for rule in SAFETY_RULES)
    + "\n\nFormato da resposta: texto objetivo em português do Brasil, citando "
    "as fontes entre parênteses no formato (Fonte: ...).\n"
    "Responda DIRETAMENTE à pergunta do profissional. Use o prontuário e os "
    "protocolos SOMENTE se forem relevantes para aquela pergunta."
)

# ── Identidade / capacidades ───────────────────────────────────────────────

_META_QUESTION = re.compile(
    r"(?i)\b("
    r"quem\s+[ée]\s+voc[eê]|o\s+que\s+(?:voc[eê]\s+)?(?:pode|consegue)\s+fazer|"
    r"o\s+que\s+voc[eê]\s+[ée]|quais?\s+(?:suas?\s+)?(?:fun[çc][õo]es|capacidades)|"
    r"apresente-se|se\s+apresente|como\s+voc[eê]\s+funciona|"
    r"what\s+can\s+you\s+do|who\s+are\s+you"
    r")\b"
)

IDENTITY_RESPONSE = (
    "Sou o assistente médico virtual do hospital, treinado com protocolos "
    "internos e literatura médica (PubMedQA) via fine-tuning LoRA.\n\n"
    "Posso ajudar a equipe de saúde a:\n"
    "- esclarecer protocolos institucionais e FAQs clínicas;\n"
    "- consultar o prontuário estruturado (dados anonimizados) e listar exames pendentes;\n"
    "- contextualizar respostas com o paciente informado e citar as fontes usadas;\n"
    "- emitir alertas para a equipe (ex.: resultado crítico, exame pendente).\n\n"
    "Não posso: prescrever medicamentos, definir posologia, dar diagnóstico "
    "definitivo nem expor dados pessoais identificáveis. Toda conduta exige "
    "validação humana. (Fonte: politica_seguranca)\n\n"
    f"{DISCLAIMER}"
)

# ── Fora de escopo (não clínico) ───────────────────────────────────────────

_OUT_OF_SCOPE_PATTERNS = re.compile(
    r"(?i)\b("
    r"que\s+dia\s+(?:[ée]|estamos|foi|ser[aá])|"
    r"que\s+horas?|"
    r"qual\s+(?:a\s+)?data\s+de\s+hoje|"
    r"como\s+est[aá]\s+o\s+tempo|"
    r"previs[aã]o\s+do\s+tempo|clima|"
    r"conte\s+(?:uma\s+)?piada|me\s+conta\s+(?:uma\s+)?piada|"
    r"futebol|filme|m[uú]sica|receita\s+de\s+bolo|"
    r"bate[\s-]?papo|conversar\s+comigo|"
    r"quem\s+ganhou|capital\s+do\s+brasil|"
    r"me\s+divirta|conte\s+(?:uma\s+)?hist[oó]ria"
    r")\b"
)

# Sinais de que a pergunta é clínica / alinhada ao desafio.
_CLINICAL_SIGNALS = re.compile(
    r"(?i)\b("
    r"protocolo|exame|paciente|prontu[aá]rio|alerta|conduta|tratamento|"
    r"sepse|avc|trombol|bi[\s-]?rads|inr|lactato|mamograf|antibiot|"
    r"alergia|hipoglicemia|pot[aá]ssio|contraste|cirurgia|uti|"
    r"hemostasia|troponina|ecg|hemocultura|glicemia|creatinina|"
    r"perfil\s+renal|te\s*v|profilaxia|analgesia|dor\s+tor[aá]cica|"
    r"n[oó]dulo|bi[oó]psia|anticoagul|pneumonia|ic\b|fibrila|"
    r"faq|fonte|validação\s+humana|plantonista|mastolog|"
    r"pendente|cr[ií]tico|resultado"
    r")\b"
)

OUT_OF_SCOPE_RESPONSE = (
    "Esta pergunta está fora do meu escopo. Sou um assistente clínico do "
    "hospital: ajudo com protocolos internos, FAQs médicas, contextualização "
    "de prontuário e alertas para a equipe — não faço conversa geral "
    "(data, clima, entretenimento etc.).\n\n"
    "Exemplos do que posso responder:\n"
    '- "Qual o protocolo de sepse na primeira hora?"\n'
    '- "Qual o próximo passo para BI-RADS 4?" (com --paciente PAC-001)\n'
    '- "Como conduzir paciente com INR elevado?" (com --paciente PAC-002)\n\n'
    "(Fonte: politica_seguranca)\n\n"
    f"{DISCLAIMER}"
)


def is_meta_question(question: str) -> bool:
    """True se a pergunta é sobre o próprio assistente (identidade/capacidades)."""
    return bool(_META_QUESTION.search(question or ""))


def is_out_of_scope_question(question: str) -> bool:
    """True se a pergunta não é clínica (chat geral / trivial).

    Duas regras:
    1. Padrões explícitos (data, clima, piada...);
    2. Nenhum sinal clínico detectado na pergunta.
    """
    text = (question or "").strip()
    if not text:
        return True
    if is_meta_question(text):
        return False
    if _OUT_OF_SCOPE_PATTERNS.search(text):
        return True
    if not _CLINICAL_SIGNALS.search(text):
        return True
    return False


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
        "Responda DIRETAMENTE à pergunta acima. Use o contexto do paciente e os "
        "protocolos apenas se forem relevantes. Cite as fontes usadas. "
        "Não prescreva; não dê diagnóstico definitivo; indique validação humana."
    )
