"""Dados sintéticos internos do hospital (requisitos R1 e R12).

Gera três tipos de documento exigidos pelo enunciado:
- Protocolos médicos internos;
- Perguntas frequentes (FAQ) feitas por médicos;
- Modelos de laudos, receitas e procedimentos internos.

Todos os dados são fictícios (hospital "Hospital Vida Plena", IDs sintéticos)
e passam por anonimização antes de virar dataset de treino.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from assistente_medico.config import HOSPITAL_DATA_DIR

HOSPITAL_NAME = "Hospital Vida Plena (fictício)"

# ── Protocolos internos ────────────────────────────────────────────────────

PROTOCOLS: list[dict[str, Any]] = [
    {
        "id": "PROTO-001",
        "titulo": "Protocolo de Sepse — Primeira Hora",
        "categoria": "emergencia",
        "conteudo": (
            "Diante de suspeita de sepse (qSOFA >= 2 ou disfunção orgânica aguda): "
            "1) Coletar lactato sérico e duas hemoculturas antes do antibiótico; "
            "2) Iniciar antibioticoterapia empírica de amplo espectro em até 60 minutos, "
            "conforme prescrição médica; 3) Reposição volêmica com cristaloide 30 mL/kg "
            "se hipotensão ou lactato >= 4 mmol/L; 4) Reavaliar em 1 hora; acionar UTI se "
            "PAM < 65 mmHg após volume. Toda conduta exige validação do médico plantonista."
        ),
    },
    {
        "id": "PROTO-002",
        "titulo": "Protocolo de Dor Torácica na Emergência",
        "categoria": "emergencia",
        "conteudo": (
            "Paciente com dor torácica: ECG de 12 derivações em até 10 minutos da chegada. "
            "Troponina na admissão e em 3 horas. Estratificar risco com escore HEART: "
            "0-3 baixo risco, 4-6 intermediário (observação e troponina seriada), "
            ">= 7 alto risco (acionar cardiologia). Supra de ST: ativar protocolo de "
            "reperfusão imediatamente. Decisão terapêutica é exclusiva do médico assistente."
        ),
    },
    {
        "id": "PROTO-003",
        "titulo": "Protocolo de Profilaxia de Tromboembolismo Venoso (TEV)",
        "categoria": "internacao",
        "conteudo": (
            "Avaliar risco de TEV com escore de Pádua em todo paciente clínico internado. "
            "Pádua >= 4: indicada profilaxia farmacológica, salvo contraindicação "
            "(sangramento ativo, plaquetas < 50.000). Contraindicação farmacológica: usar "
            "profilaxia mecânica (compressão pneumática). Reavaliar risco a cada 48 horas. "
            "A escolha e dose do anticoagulante são de responsabilidade do prescritor."
        ),
    },
    {
        "id": "PROTO-004",
        "titulo": "Protocolo de Hipoglicemia em Paciente Internado",
        "categoria": "internacao",
        "conteudo": (
            "Glicemia < 70 mg/dL: paciente consciente e com via oral liberada, ofertar 15 g "
            "de carboidrato rápido e repetir glicemia capilar em 15 minutos. Paciente "
            "inconsciente ou sem via oral: glicose hipertônica EV conforme prescrição. "
            "Repetir até glicemia > 70 mg/dL, depois refeição completa. Notificar médico "
            "responsável e revisar esquema de insulina/hipoglicemiantes."
        ),
    },
    {
        "id": "PROTO-005",
        "titulo": "Protocolo de Triagem de Câncer de Mama",
        "categoria": "rastreamento",
        "conteudo": (
            "Mamografia bienal para mulheres de 50 a 69 anos (rastreamento populacional). "
            "Alto risco (BRCA1/2, história familiar de 1º grau precoce): iniciar rastreio "
            "mais cedo com avaliação do mastologista. BI-RADS 4 ou 5: encaminhar para "
            "biópsia. Resultados de modelos preditivos de triagem são apoio à decisão e "
            "nunca substituem confirmação histopatológica."
        ),
    },
    {
        "id": "PROTO-006",
        "titulo": "Protocolo de Antibioticoprofilaxia Cirúrgica",
        "categoria": "cirurgia",
        "conteudo": (
            "Administrar antibiótico profilático em até 60 minutos antes da incisão "
            "cirúrgica, conforme padronização institucional por sítio cirúrgico. "
            "Dose adicional se cirurgia > 4 horas ou sangramento > 1500 mL. "
            "Suspender profilaxia em até 24 horas no pós-operatório na maioria dos casos. "
            "Alergia a betalactâmicos deve ser registrada e alternativa definida pelo prescritor."
        ),
    },
    {
        "id": "PROTO-007",
        "titulo": "Protocolo de Prevenção de Lesão Renal Aguda por Contraste",
        "categoria": "exames",
        "conteudo": (
            "Pacientes com TFG < 45 mL/min/1,73m² em exame contrastado eletivo: hidratação "
            "venosa com solução salina isotônica antes e após o exame; suspender "
            "temporariamente metformina e reintroduzir após 48 horas com função renal "
            "reavaliada, conforme decisão médica. Dosar creatinina 48-72 horas após contraste."
        ),
    },
    {
        "id": "PROTO-008",
        "titulo": "Protocolo de Manejo Inicial de AVC Isquêmico",
        "categoria": "emergencia",
        "conteudo": (
            "Suspeita de AVC: aplicar escala NIHSS e TC de crânio sem contraste em até "
            "25 minutos da chegada. Janela para trombólise EV: até 4,5 horas do início dos "
            "sintomas, após exclusão de hemorragia e contraindicações. Glicemia capilar "
            "imediata para excluir hipoglicemia. Ativação do time de AVC é obrigatória; "
            "a indicação de trombólise é decisão exclusiva do neurologista."
        ),
    },
    {
        "id": "PROTO-009",
        "titulo": "Protocolo de Alta Responsável",
        "categoria": "internacao",
        "conteudo": (
            "Antes da alta: conferir exames pendentes e resultados críticos; conciliar "
            "medicamentos de uso contínuo; entregar receita e orientações por escrito; "
            "agendar retorno ambulatorial quando indicado. Pendências de exame com "
            "potencial de mudar conduta impedem alta até avaliação médica documentada."
        ),
    },
    {
        "id": "PROTO-010",
        "titulo": "Protocolo de Analgesia Escalonada",
        "categoria": "internacao",
        "conteudo": (
            "Avaliar dor com escala numérica (0-10) a cada aferição de sinais vitais. "
            "Dor leve (1-3): analgésicos simples conforme prescrição. Dor moderada (4-6): "
            "associar anti-inflamatório se não houver contraindicação. Dor intensa (7-10): "
            "comunicar médico para avaliação de opioide. Reavaliar 30-60 minutos após "
            "medicação. Toda prescrição é ato médico e exige validação humana."
        ),
    },
    {
        "id": "PROTO-011",
        "titulo": "Protocolo de Alerta de Resultados Críticos de Laboratório",
        "categoria": "exames",
        "conteudo": (
            "Valores críticos (ex.: potássio < 2,5 ou > 6,5 mEq/L; glicemia < 45 mg/dL; "
            "hemoglobina < 6,5 g/dL; INR > 5 em anticoagulado) devem ser comunicados ao "
            "médico responsável em até 30 minutos, com registro de quem recebeu. O sistema "
            "de apoio deve gerar alerta automático para a equipe assistente."
        ),
    },
    {
        "id": "PROTO-012",
        "titulo": "Protocolo de Reconciliação Medicamentosa e Alergias",
        "categoria": "seguranca",
        "conteudo": (
            "Registrar alergias medicamentosas na admissão em campo estruturado. "
            "Prescrições que envolvam classe com alergia registrada devem gerar bloqueio e "
            "dupla checagem. Qualquer sugestão automatizada de tratamento deve verificar a "
            "lista de alergias do paciente e sinalizar conflito antes de chegar ao prescritor."
        ),
    },
]

# ── FAQ de médicos ─────────────────────────────────────────────────────────

FAQS: list[dict[str, str]] = [
    {
        "pergunta": "Qual o tempo máximo para iniciar antibiótico na sepse?",
        "resposta": (
            "Pelo protocolo interno PROTO-001, a antibioticoterapia empírica deve ser "
            "iniciada em até 60 minutos da suspeita de sepse, após coleta de lactato e "
            "hemoculturas. A escolha do esquema é do médico plantonista."
        ),
        "fonte": "PROTO-001",
    },
    {
        "pergunta": "Quando o escore HEART indica acionamento da cardiologia?",
        "resposta": (
            "Escore HEART >= 7 caracteriza alto risco e indica acionamento imediato da "
            "cardiologia, conforme PROTO-002. Entre 4 e 6, manter observação com troponina seriada."
        ),
        "fonte": "PROTO-002",
    },
    {
        "pergunta": "Quando indicar profilaxia farmacológica de TEV em paciente clínico?",
        "resposta": (
            "Escore de Pádua >= 4 indica profilaxia farmacológica, salvo contraindicações "
            "como sangramento ativo ou plaquetas < 50.000 (PROTO-003). Nesses casos, usar "
            "profilaxia mecânica."
        ),
        "fonte": "PROTO-003",
    },
    {
        "pergunta": "Como conduzir hipoglicemia em paciente consciente internado?",
        "resposta": (
            "Ofertar 15 g de carboidrato de ação rápida por via oral e repetir a glicemia "
            "capilar em 15 minutos, repetindo o ciclo até > 70 mg/dL (PROTO-004). Notificar "
            "o médico responsável para revisão do esquema hipoglicemiante."
        ),
        "fonte": "PROTO-004",
    },
    {
        "pergunta": "Qual a faixa etária do rastreamento mamográfico populacional?",
        "resposta": (
            "Mamografia bienal para mulheres de 50 a 69 anos, conforme PROTO-005. Pacientes "
            "de alto risco devem ser avaliadas individualmente pelo mastologista."
        ),
        "fonte": "PROTO-005",
    },
    {
        "pergunta": "O resultado BI-RADS 4 obriga biópsia?",
        "resposta": (
            "BI-RADS 4 ou 5 indica encaminhamento para biópsia segundo o PROTO-005. "
            "A confirmação diagnóstica é sempre histopatológica."
        ),
        "fonte": "PROTO-005",
    },
    {
        "pergunta": "Quando administrar a antibioticoprofilaxia cirúrgica?",
        "resposta": (
            "Em até 60 minutos antes da incisão, com dose adicional se a cirurgia durar "
            "mais de 4 horas ou houver sangramento > 1500 mL (PROTO-006)."
        ),
        "fonte": "PROTO-006",
    },
    {
        "pergunta": "Preciso suspender metformina antes de exame com contraste?",
        "resposta": (
            "Em pacientes com TFG < 45, o PROTO-007 orienta suspensão temporária da "
            "metformina, com reintrodução após 48 horas mediante reavaliação da função "
            "renal pelo médico assistente."
        ),
        "fonte": "PROTO-007",
    },
    {
        "pergunta": "Qual a janela para trombólise no AVC isquêmico?",
        "resposta": (
            "Até 4,5 horas do início dos sintomas, após TC excluir hemorragia e revisão "
            "de contraindicações (PROTO-008). A indicação é decisão do neurologista."
        ),
        "fonte": "PROTO-008",
    },
    {
        "pergunta": "Posso dar alta com exame pendente?",
        "resposta": (
            "Não, se a pendência tiver potencial de mudar conduta: o PROTO-009 exige "
            "avaliação médica documentada das pendências antes da alta responsável."
        ),
        "fonte": "PROTO-009",
    },
    {
        "pergunta": "Como escalonar analgesia para dor intensa?",
        "resposta": (
            "Dor 7-10 na escala numérica: comunicar o médico para avaliação de opioide e "
            "reavaliar em 30-60 minutos após a medicação (PROTO-010). O assistente não "
            "define fármaco nem dose."
        ),
        "fonte": "PROTO-010",
    },
    {
        "pergunta": "Qual o prazo para comunicar um potássio de 6,8 mEq/L?",
        "resposta": (
            "Potássio > 6,5 mEq/L é valor crítico: comunicação ao médico responsável em "
            "até 30 minutos com registro do receptor, conforme PROTO-011."
        ),
        "fonte": "PROTO-011",
    },
    {
        "pergunta": "O que fazer se a sugestão de tratamento conflita com alergia registrada?",
        "resposta": (
            "O PROTO-012 determina bloqueio e dupla checagem: o conflito deve ser "
            "sinalizado ao prescritor antes de qualquer decisão. Sistemas automatizados "
            "nunca prescrevem; apenas alertam."
        ),
        "fonte": "PROTO-012",
    },
    {
        "pergunta": "O assistente virtual pode prescrever um antibiótico direto para o paciente?",
        "resposta": (
            "Não. Por limite de segurança institucional, o assistente nunca prescreve "
            "medicamentos nem define posologia. Ele pode indicar o protocolo aplicável e "
            "sugerir avaliação, mas toda prescrição exige validação humana de um médico."
        ),
        "fonte": "politica_seguranca",
    },
    {
        "pergunta": "O assistente pode dar diagnóstico definitivo?",
        "resposta": (
            "Não. O assistente oferece apoio à decisão com base em protocolos e literatura, "
            "citando fontes, mas o diagnóstico definitivo é ato médico que exige avaliação "
            "clínica humana."
        ),
        "fonte": "politica_seguranca",
    },
]

# ── Modelos de laudo, receita e procedimento ───────────────────────────────

TEMPLATES: list[dict[str, str]] = [
    {
        "id": "TPL-LAUDO-001",
        "tipo": "laudo",
        "titulo": "Modelo de laudo de mamografia",
        "conteudo": (
            "LAUDO DE MAMOGRAFIA — {hospital}\n"
            "Identificação: [NOME_REMOVIDO], registro [PRONTUARIO_REMOVIDO].\n"
            "Indicação: rastreamento.\n"
            "Achados: parênquima mamário de densidade {densidade}. {achados}\n"
            "Classificação: BI-RADS {birads}.\n"
            "Conclusão/Recomendação: {recomendacao}.\n"
            "Este laudo deve ser interpretado pelo médico solicitante."
        ),
    },
    {
        "id": "TPL-LAUDO-002",
        "tipo": "laudo",
        "titulo": "Modelo de laudo de tomografia de crânio",
        "conteudo": (
            "LAUDO DE TC DE CRÂNIO SEM CONTRASTE — {hospital}\n"
            "Identificação: [NOME_REMOVIDO].\n"
            "Técnica: cortes axiais sem contraste.\n"
            "Achados: {achados}\n"
            "Impressão: {impressao}.\n"
            "Correlação clínica é indispensável."
        ),
    },
    {
        "id": "TPL-RECEITA-001",
        "tipo": "receita",
        "titulo": "Modelo de receita ambulatorial",
        "conteudo": (
            "RECEITUÁRIO — {hospital}\n"
            "Paciente: [NOME_REMOVIDO]\n"
            "Uso {via}:\n1. {medicamento} — {posologia} — {duracao}.\n"
            "Orientações: {orientacoes}\n"
            "Assinatura e carimbo do médico prescritor (obrigatório): ______________\n"
            "Documento inválido sem assinatura do profissional habilitado."
        ),
    },
    {
        "id": "TPL-PROC-001",
        "tipo": "procedimento",
        "titulo": "Checklist de procedimento — passagem de cateter venoso central",
        "conteudo": (
            "PROCEDIMENTO: passagem de CVC — {hospital}\n"
            "1. Confirmar indicação e consentimento informado;\n"
            "2. Higiene das mãos e paramentação estéril completa;\n"
            "3. Antissepsia com clorexidina alcoólica;\n"
            "4. Punção guiada por ultrassom quando disponível;\n"
            "5. Confirmar posicionamento com radiografia antes do uso;\n"
            "6. Registrar procedimento e intercorrências em prontuário."
        ),
    },
    {
        "id": "TPL-PROC-002",
        "tipo": "procedimento",
        "titulo": "Checklist de cirurgia segura (adaptado OMS)",
        "conteudo": (
            "CHECKLIST DE CIRURGIA SEGURA — {hospital}\n"
            "Antes da indução: confirmar identidade, sítio cirúrgico, procedimento e alergias.\n"
            "Antes da incisão: time-out com toda a equipe; confirmar antibioticoprofilaxia "
            "(PROTO-006).\n"
            "Antes da saída da sala: contagem de compressas e instrumentais; identificar "
            "amostras; registrar pendências para recuperação."
        ),
    },
]


def _protocol_training_examples() -> list[dict[str, str]]:
    examples = []
    for proto in PROTOCOLS:
        examples.append(
            {
                "instruction": f"Segundo o protocolo interno do hospital, {proto['titulo'].lower()}: o que orienta?",
                "output": f"{proto['conteudo']} (Fonte: {proto['id']} — {proto['titulo']})",
                "origem": f"protocolo:{proto['id']}",
            }
        )
    return examples


def _faq_training_examples() -> list[dict[str, str]]:
    return [
        {
            "instruction": faq["pergunta"],
            "output": f"{faq['resposta']} (Fonte: {faq['fonte']})",
            "origem": f"faq:{faq['fonte']}",
        }
        for faq in FAQS
    ]


def _template_training_examples() -> list[dict[str, str]]:
    examples = []
    for tpl in TEMPLATES:
        examples.append(
            {
                "instruction": f"Qual o modelo institucional de {tpl['titulo'].lower()}?",
                "output": f"{tpl['conteudo']} (Fonte: {tpl['id']})",
                "origem": f"template:{tpl['id']}",
            }
        )
    return examples


# Exemplos de segurança: ensinam o modelo a recusar prescrição/diagnóstico direto.
SAFETY_EXAMPLES: list[dict[str, str]] = [
    {
        "instruction": "Prescreva amoxicilina 500mg de 8 em 8 horas para meu paciente com sinusite.",
        "output": (
            "Não posso prescrever medicamentos nem definir posologia: prescrição é ato "
            "médico que exige validação humana. Posso apoiar indicando o protocolo "
            "institucional aplicável e os pontos de verificação (alergias registradas, "
            "função renal). A decisão final é do médico assistente. (Fonte: politica_seguranca)"
        ),
        "origem": "seguranca",
    },
    {
        "instruction": "Me diga o diagnóstico definitivo deste paciente com dor torácica.",
        "output": (
            "Não forneço diagnóstico definitivo. Posso ajudar com a estratificação "
            "sugerida pelo protocolo de dor torácica (ECG em 10 minutos, troponina "
            "seriada, escore HEART — PROTO-002), mas o diagnóstico é ato médico que "
            "requer avaliação clínica humana. (Fonte: PROTO-002; politica_seguranca)"
        ),
        "origem": "seguranca",
    },
    {
        "instruction": "Qual o CPF e endereço do paciente do leito 12?",
        "output": (
            "Não posso fornecer dados pessoais identificáveis de pacientes. O acesso a "
            "esses dados é restrito e auditado conforme a política de privacidade do "
            "hospital. Posso ajudar com informações clínicas não identificáveis "
            "necessárias à conduta. (Fonte: politica_seguranca)"
        ),
        "origem": "seguranca",
    },
    {
        "instruction": "Aumente a dose do anticoagulante do paciente, ele está com INR baixo.",
        "output": (
            "Ajuste de dose de anticoagulante é decisão exclusiva do prescritor. Posso "
            "sinalizar o resultado ao médico responsável e lembrar o protocolo de valores "
            "críticos (INR > 5 exige comunicação em 30 minutos — PROTO-011). Nenhuma "
            "alteração de prescrição é feita sem validação humana. (Fonte: PROTO-011; "
            "politica_seguranca)"
        ),
        "origem": "seguranca",
    },
]


def build_hospital_training_examples() -> list[dict[str, str]]:
    """Todos os exemplos de treino derivados dos dados internos sintéticos."""
    return (
        _protocol_training_examples()
        + _faq_training_examples()
        + _template_training_examples()
        + SAFETY_EXAMPLES
    )


def export_hospital_documents(output_dir: Path | None = None) -> dict[str, Path]:
    """Salva os documentos sintéticos em data/hospital/ (dataset versionável)."""
    out_dir = output_dir or HOSPITAL_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, payload in (
        ("protocolos.json", PROTOCOLS),
        ("faq_medicos.json", FAQS),
        ("templates_documentos.json", TEMPLATES),
    ):
        path = out_dir / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        paths[name] = path
    return paths
