"""Testes da base de conhecimento e das tools LangChain."""

import json

import pytest

from assistente_medico.assistant.retrieval import KnowledgeBase, format_context_docs
from assistente_medico.assistant.tools import build_tools
from assistente_medico.db.repository import HospitalRepository
from assistente_medico.db.seed_patients import seed_database


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase()


def test_busca_protocolo_sepse(kb):
    docs = kb.search("protocolo de sepse primeira hora antibiótico")
    assert docs
    assert any("PROTO-001" in d["ref"] for d in docs)


def test_busca_avc(kb):
    docs = kb.search("janela trombólise AVC isquêmico")
    assert any("PROTO-008" in d["ref"] for d in docs)


def test_format_context_cita_fontes(kb):
    docs = kb.search("rastreamento mamografia")
    text = format_context_docs(docs)
    assert "(Fonte:" in text


def test_tools_langchain(tmp_path):
    repo = HospitalRepository(db_path=tmp_path / "test.db")
    seed_database(repo=repo, export_json=False)
    tools = build_tools(repo, KnowledgeBase())
    by_name = {t.name: t for t in tools}
    assert {
        "buscar_paciente",
        "listar_exames_pendentes",
        "listar_resultados_criticos",
        "buscar_protocolo",
        "registrar_alerta",
    } <= set(by_name)

    paciente = json.loads(by_name["buscar_paciente"].invoke({"paciente_id": "PAC-001"}))
    assert paciente["id"] == "PAC-001"

    pendentes = json.loads(by_name["listar_exames_pendentes"].invoke({"paciente_id": "PAC-001"}))
    assert isinstance(pendentes, list) and pendentes

    resultado = json.loads(
        by_name["registrar_alerta"].invoke(
            {"paciente_id": "PAC-001", "tipo": "teste", "mensagem": "alerta de teste"}
        )
    )
    assert resultado["status"] == "registrado"
    repo.close()
