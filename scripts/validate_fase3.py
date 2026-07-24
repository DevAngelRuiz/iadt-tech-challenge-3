"""Validação dos critérios de aceitação da Fase 3 (R1–R12).

Verifica que cada requisito do enunciado tem artefato/código correspondente
e que os fluxos críticos funcionam. Sai com código != 0 se algo falhar.

Uso:
    .venv/bin/python scripts/validate_fase3.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS: list[tuple[str, str, bool, str]] = []


def check(req: str, descricao: str, ok: bool, detalhe: str = "") -> None:
    RESULTS.append((req, descricao, ok, detalhe))


def main() -> int:
    # R1 — Pipeline de fine-tuning
    train_module = ROOT / "assistente_medico/fine_tuning/train_lora.py"
    check("R1", "Pipeline de fine-tuning LoRA (MLX)", train_module.exists(), str(train_module))
    adapter = ROOT / "models/fase3_lora/adapters.safetensors"
    check("R1", "Adapter LoRA treinado", adapter.exists(),
          str(adapter) if adapter.exists() else "rode: python main_fase3.py train")

    # R2 — Preprocessing, anonimização e curadoria
    from assistente_medico.data.anonymize import anonymize_text

    res = anonymize_text("Paciente João da Silva, CPF 123.456.789-01")
    check("R2", "Anonimização remove PII", res.total_replacements >= 2, f"{res.replacements}")

    # R3/R5 — Pipeline LangChain com LLM customizada + contexto do paciente
    from assistente_medico.assistant.graph import MedicalAssistantGraph
    from assistente_medico.assistant.llm import AssistantLLM
    from assistente_medico.assistant.logging_audit import AuditLogger
    from assistente_medico.db.repository import HospitalRepository
    from assistente_medico.db.seed_patients import seed_database

    import tempfile

    tmp = Path(tempfile.mkdtemp())
    repo = HospitalRepository(db_path=tmp / "val.db")
    seed_database(repo=repo, export_json=False)
    llm = AssistantLLM(config={"backend": "stub", "model": "offline_stub", "temperature": 0.1})
    audit = AuditLogger(log_path=tmp / "audit.jsonl")
    assistant = MedicalAssistantGraph(repo=repo, llm=llm, audit=audit)

    r = assistant.ask("Qual o próximo passo para paciente com BI-RADS 4?", "PAC-001")
    check("R3", "Pipeline LangChain/LangGraph responde", bool(r["resposta"]))
    check("R5", "Resposta contextualizada com dados do paciente",
          any(f.startswith("prontuario:PAC-001") for f in r["fontes"]), str(r["fontes"]))

    # R4 — Consultas a base estruturada
    from assistente_medico.assistant.retrieval import KnowledgeBase
    from assistente_medico.assistant.tools import build_tools

    tools = build_tools(repo, KnowledgeBase())
    names = {t.name for t in tools}
    check("R4", "Tools LangChain de consulta estruturada (SQLite)",
          {"buscar_paciente", "listar_exames_pendentes", "registrar_alerta"} <= names, str(names))

    # R6 — Limites de atuação
    blocked = assistant.ask("Prescreva amoxicilina 500mg de 8/8h.", "PAC-001")
    check("R6", "Guardrail recusa prescrição direta", blocked["bloqueada_pelo_guardrail"])
    check("R6", "Toda resposta exige validação humana",
          r["requer_validacao_humana"] and blocked["requer_validacao_humana"])

    # R7 — Logging para auditoria
    entries = audit.read_all()
    ok_audit = len(entries) >= 2 and all(
        {"timestamp", "pergunta", "resposta_final", "fontes", "etapas"} <= set(e) for e in entries
    )
    check("R7", "Audit log JSONL detalhado", ok_audit, f"{len(entries)} entradas")

    # R8 — Explainability (fontes)
    check("R8", "Respostas citam fontes rastreáveis", bool(r["fontes"]), str(r["fontes"][:3]))

    # R9 — Fluxo LangGraph com alertas
    critical = assistant.ask("Como conduzir paciente com INR elevado?", "PAC-002")
    check("R9", "LangGraph emite alerta para resultado crítico",
          any(a["tipo"] == "resultado_critico" for a in critical["alertas"]), str(critical["alertas"]))
    check("R9", "Fluxo executa todas as etapas do grafo", len(critical["etapas"]) >= 7,
          f"{len(critical['etapas'])} etapas")

    # R10 — Projeto modular + README
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    check("R10", "README documenta a Fase 3", "Fase 3" in readme and "main_fase3" in readme)
    check("R10", "Projeto modularizado", (ROOT / "assistente_medico/assistant/graph.py").exists())

    # R11 — Relatório técnico
    relatorio = ROOT / "docs/relatorio_tecnico_fase3.md"
    ok_rel = relatorio.exists()
    if ok_rel:
        texto = relatorio.read_text(encoding="utf-8")
        ok_rel = all(t in texto for t in ("fine-tuning", "LangGraph", "mermaid"))
    check("R11", "Relatório técnico com diagrama e avaliação", ok_rel, str(relatorio))

    # R12 — Dataset anonimizado/sintético no repositório
    for name, path in (
        ("Protocolos sintéticos", ROOT / "data/hospital/protocolos.json"),
        ("Pacientes sintéticos", ROOT / "data/patients/pacientes_sinteticos.json"),
        ("Dataset de treino (JSONL)", ROOT / "data/training/train.jsonl"),
    ):
        check("R12", name, path.exists(), str(path))

    repo.close()

    # ── Relatório final ────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("VALIDAÇÃO DOS CRITÉRIOS DE ACEITAÇÃO — FASE 3")
    print("=" * 78)
    failed = 0
    for req, desc, ok, detail in RESULTS:
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        line = f"[{status}] {req}: {desc}"
        if detail and not ok:
            line += f"  -> {detail}"
        print(line)
    total = len(RESULTS)
    print("-" * 78)
    print(f"Resultado: {total - failed}/{total} verificações aprovadas")
    if failed:
        print("ATENÇÃO: há critérios pendentes acima (FAIL).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
