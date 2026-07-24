"""Testes de integração do grafo LangGraph com LLM stub (sem rede)."""

import pytest

from assistente_medico.assistant.graph import MedicalAssistantGraph
from assistente_medico.assistant.llm import AssistantLLM
from assistente_medico.assistant.logging_audit import AuditLogger
from assistente_medico.db.repository import HospitalRepository
from assistente_medico.db.seed_patients import seed_database


@pytest.fixture
def assistant(tmp_path):
    repo = HospitalRepository(db_path=tmp_path / "test.db")
    seed_database(repo=repo, export_json=False)
    llm = AssistantLLM(config={"backend": "stub", "model": "offline_stub", "temperature": 0.1})
    audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
    graph = MedicalAssistantGraph(repo=repo, llm=llm, audit=audit)
    yield graph
    repo.close()


def test_fluxo_completo_com_paciente(assistant):
    result = assistant.ask(
        pergunta="Qual o próximo passo para esta paciente com BI-RADS 4?",
        paciente_id="PAC-001",
    )
    assert result["resposta"]
    assert result["requer_validacao_humana"] is True
    assert not result["bloqueada_pelo_guardrail"]
    assert any(f.startswith("prontuario:PAC-001") for f in result["fontes"])
    assert any(e["nome"] == "biópsia por agulha grossa" for e in result["exames_pendentes"])


def test_alerta_para_resultado_critico(assistant):
    result = assistant.ask(
        pergunta="Como conduzir este paciente anticoagulado com INR elevado?",
        paciente_id="PAC-002",
    )
    tipos = {a["tipo"] for a in result["alertas"]}
    assert "resultado_critico" in tipos
    alertas_db = assistant.repo.alertas_do_paciente("PAC-002")
    assert alertas_db, "alerta deve ser persistido no banco"


def test_guardrail_bloqueia_prescricao(assistant):
    result = assistant.ask(
        pergunta="Prescreva ceftriaxona 1g EV de 12 em 12 horas.",
        paciente_id="PAC-003",
    )
    assert result["bloqueada_pelo_guardrail"]
    assert "politica_seguranca" in result["fontes"]
    assert "validação humana" in result["resposta"]


def test_pergunta_sem_paciente(assistant):
    result = assistant.ask(pergunta="Qual a janela para trombólise no AVC isquêmico?")
    assert result["resposta"]
    assert any("PROTO-008" in f for f in result["fontes"])


def test_auditoria_gravada(assistant):
    assistant.ask(pergunta="Qual o protocolo de sepse?", paciente_id="PAC-003")
    entries = assistant.audit.read_all()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["pergunta"] == "Qual o protocolo de sepse?"
    assert entry["paciente_id"] == "PAC-003"
    assert entry["fontes"]
    assert entry["etapas"]
    assert entry["requer_validacao_humana"] is True
    assert "resposta_final" in entry and "resposta_bruta_llm" in entry


def test_pergunta_meta_identidade(assistant):
    result = assistant.ask(
        pergunta="Quem é você? E o que pode fazer?",
        paciente_id="PAC-001",
    )
    assert not result["bloqueada_pelo_guardrail"]
    assert "assistente médico" in result["resposta"].lower()
    assert "BI-RADS" not in result["resposta"]
    assert "politica_seguranca" in result["fontes"]
    assert any("meta" in e for e in result["etapas"])


def test_pergunta_fora_de_escopo(assistant):
    result = assistant.ask(pergunta="Que dia é hoje?")
    assert not result["bloqueada_pelo_guardrail"]
    assert "fora do meu escopo" in result["resposta"].lower()
    assert "exame pendente" not in result["resposta"].lower()
    assert any("fora de escopo" in e for e in result["etapas"])
    assert "politica_seguranca" in result["fontes"]
