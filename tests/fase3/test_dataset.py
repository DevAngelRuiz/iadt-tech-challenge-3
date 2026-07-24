"""Testes do pipeline de dados: curadoria, anonimização e formato de treino."""

import json

from assistente_medico.data.prepare_dataset import SYSTEM_PROMPT, _to_chat_record
from assistente_medico.data.synthetic_hospital import (
    FAQS,
    PROTOCOLS,
    SAFETY_EXAMPLES,
    TEMPLATES,
    build_hospital_training_examples,
)


def test_exemplos_hospitalares_cobrem_todos_os_tipos():
    examples = build_hospital_training_examples()
    origens = {e["origem"].split(":")[0] for e in examples}
    assert {"protocolo", "faq", "template", "seguranca"} <= origens
    assert len(examples) == len(PROTOCOLS) + len(FAQS) + len(TEMPLATES) + len(SAFETY_EXAMPLES)


def test_chat_record_formato_mlx():
    example = {"instruction": "Qual o protocolo de sepse?", "output": "Ver PROTO-001.", "origem": "faq:PROTO-001"}
    record = _to_chat_record(example)
    roles = [m["role"] for m in record["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert record["messages"][0]["content"] == SYSTEM_PROMPT
    assert record["origem"] == "faq:PROTO-001"


def test_chat_record_anonimiza_pii():
    example = {
        "instruction": "O paciente José Almeida, CPF 123.456.789-01, pode receber contraste?",
        "output": "Avaliar função renal (Fonte: PROTO-007).",
        "origem": "faq:PROTO-007",
    }
    record = _to_chat_record(example)
    user_msg = record["messages"][1]["content"]
    assert "123.456.789-01" not in user_msg
    assert "José Almeida" not in user_msg


def test_exemplos_de_seguranca_recusam():
    refusal_markers = ("não posso", "não forneço", "validação humana", "decisão exclusiva")
    for ex in SAFETY_EXAMPLES:
        output = ex["output"].lower()
        assert any(marker in output for marker in refusal_markers)
        assert "Fonte:" in ex["output"]


def test_dataset_gerado_e_valido(tmp_path):
    from assistente_medico.data.prepare_dataset import prepare_dataset

    stats = prepare_dataset(max_pubmedqa=0, output_dir=tmp_path)
    assert stats["hospital_examples"] > 0
    train_path = tmp_path / "train.jsonl"
    assert train_path.exists()
    with train_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            assert "messages" in rec
