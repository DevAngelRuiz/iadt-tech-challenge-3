"""Validação dos critérios de aceitação da Fase 3 (R1–R12).

Para cada requisito do enunciado, verifica que o artefato/código correspondente
existe e funciona, e mostra ONDE está cada arquivo — serve como guia de leitura
do projeto. Antes dos checks, imprime o status da LLM em uso (fine-tuned MLX,
Ollama ou stub offline). Sai com código != 0 se algo falhar.

Uso:
    .venv/bin/python scripts/validate_fase3.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS: list[tuple[str, str, bool, list[str], str]] = []


def check(req: str, descricao: str, ok: bool, arquivos: list[str], detalhe: str = "") -> None:
    RESULTS.append((req, descricao, ok, arquivos, detalhe))


def print_llm_status() -> bool:
    """Mostra o backend da LLM resolvido para runtime e valida a configuração."""
    from assistente_medico.config import (
        ADAPTER_DIR,
        load_project_env,
        resolve_assistant_llm_config,
    )

    load_project_env()
    cfg = resolve_assistant_llm_config()
    backend = cfg["backend"]
    adapter_file = ADAPTER_DIR / "adapters.safetensors"
    adapter_exists = adapter_file.exists()

    print("\n" + "=" * 78)
    print("STATUS DA LLM (backend resolvido para runtime)")
    print("=" * 78)
    print(f"Backend:  {backend}")
    print(f"Modelo:   {cfg['model']}")
    if backend == "mlx":
        print(f"Adapter:  {cfg.get('adapter_path') or 'NENHUM (modelo base, sem fine-tuning)'}")
        print(f"Adapter treinado existe: {'sim' if adapter_exists else 'NÃO'}")
    if backend == "ollama":
        print(f"URL:      {cfg['base_url']}")
    print("Arquivos: assistente_medico/config.py (resolve_assistant_llm_config)")
    print("          assistente_medico/assistant/llm.py (geração MLX/Ollama/stub)")
    print("          models/fase3_lora/ (adapter LoRA do fine-tuning)")

    # PASS: LLM real em uso — mlx com adapter treinado, ou ollama com modelo.
    if backend == "mlx":
        ok = adapter_exists
        veredicto = (
            "LLM fine-tuned (MLX + adapter LoRA) em uso"
            if ok
            else "backend mlx sem adapter — rode: python main_fase3.py train"
        )
    elif backend == "ollama":
        ok = bool(cfg["model"])
        veredicto = f"LLM real via Ollama ({cfg['model']})"
    else:
        ok = False
        veredicto = (
            "modo stub offline — nenhuma LLM real ativa "
            "(treine o adapter ou configure OLLAMA_MODEL no .env)"
        )
    print(f"\n[{'PASS' if ok else 'FAIL'}] {veredicto}")
    return ok


def main() -> int:
    llm_ok = print_llm_status()

    # R1 — Pipeline de fine-tuning
    check(
        "R1", "Pipeline de fine-tuning LoRA (MLX)",
        (ROOT / "assistente_medico/fine_tuning/train_lora.py").exists(),
        ["assistente_medico/fine_tuning/train_lora.py",
         "assistente_medico/fine_tuning/evaluate.py"],
    )
    adapter = ROOT / "models/fase3_lora/adapters.safetensors"
    check(
        "R1", "Adapter LoRA treinado",
        adapter.exists(),
        ["models/fase3_lora/adapters.safetensors", "outputs/fase3_train_log.txt"],
        "" if adapter.exists() else "rode: python main_fase3.py train",
    )

    # R2 — Preprocessing, anonimização e curadoria
    from assistente_medico.data.anonymize import anonymize_text

    res = anonymize_text("Paciente João da Silva, CPF 123.456.789-01")
    check(
        "R2", "Anonimização remove PII",
        res.total_replacements >= 2,
        ["assistente_medico/data/anonymize.py",
         "assistente_medico/data/prepare_dataset.py"],
        f"{res.replacements}",
    )

    # R3/R5 — Pipeline LangChain com LLM customizada + contexto do paciente
    # (checks funcionais usam o stub para rodar rápido e offline; o status da
    #  LLM real aparece na seção acima)
    from assistente_medico.assistant.graph import MedicalAssistantGraph
    from assistente_medico.assistant.llm import AssistantLLM
    from assistente_medico.assistant.logging_audit import AuditLogger
    from assistente_medico.db.repository import HospitalRepository
    from assistente_medico.db.seed_patients import seed_database

    tmp = Path(tempfile.mkdtemp())
    repo = HospitalRepository(db_path=tmp / "val.db")
    seed_database(repo=repo, export_json=False)
    llm = AssistantLLM(config={"backend": "stub", "model": "offline_stub", "temperature": 0.1})
    audit = AuditLogger(log_path=tmp / "audit.jsonl")
    assistant = MedicalAssistantGraph(repo=repo, llm=llm, audit=audit)

    r = assistant.ask("Qual o próximo passo para paciente com BI-RADS 4?", "PAC-001")
    check(
        "R3", "Pipeline LangChain/LangGraph responde",
        bool(r["resposta"]),
        ["assistente_medico/assistant/graph.py",
         "assistente_medico/assistant/llm.py",
         "assistente_medico/assistant/prompts.py"],
    )
    check(
        "R5", "Resposta contextualizada com dados do paciente",
        any(f.startswith("prontuario:PAC-001") for f in r["fontes"]),
        ["assistente_medico/assistant/graph.py (nós carregar_paciente/verificar_exames)",
         "assistente_medico/db/seed_patients.py"],
        str(r["fontes"]),
    )

    # R4 — Consultas a base estruturada
    from assistente_medico.assistant.retrieval import KnowledgeBase
    from assistente_medico.assistant.tools import build_tools

    tools = build_tools(repo, KnowledgeBase())
    names = {t.name for t in tools}
    check(
        "R4", "Tools LangChain de consulta estruturada (SQLite)",
        {"buscar_paciente", "listar_exames_pendentes", "registrar_alerta"} <= names,
        ["assistente_medico/assistant/tools.py",
         "assistente_medico/db/repository.py",
         "assistente_medico/db/schema.py"],
        str(names),
    )

    # R6 — Limites de atuação
    blocked = assistant.ask("Prescreva amoxicilina 500mg de 8/8h.", "PAC-001")
    check(
        "R6", "Guardrail recusa prescrição direta",
        blocked["bloqueada_pelo_guardrail"],
        ["assistente_medico/assistant/guardrails.py",
         "assistente_medico/config.py (SAFETY_RULES)"],
    )
    check(
        "R6", "Toda resposta exige validação humana",
        r["requer_validacao_humana"] and blocked["requer_validacao_humana"],
        ["assistente_medico/assistant/guardrails.py",
         "assistente_medico/config.py (DISCLAIMER)"],
    )

    # R7 — Logging para auditoria
    entries = audit.read_all()
    ok_audit = len(entries) >= 2 and all(
        {"timestamp", "pergunta", "resposta_final", "fontes", "etapas"} <= set(e) for e in entries
    )
    check(
        "R7", "Audit log JSONL detalhado",
        ok_audit,
        ["assistente_medico/assistant/logging_audit.py",
         "outputs/fase3_audit_log.jsonl (gerado em runtime)"],
        f"{len(entries)} entradas",
    )

    # R8 — Explainability (fontes)
    check(
        "R8", "Respostas citam fontes rastreáveis",
        bool(r["fontes"]),
        ["assistente_medico/assistant/explainability.py"],
        str(r["fontes"][:3]),
    )

    # R9 — Fluxo LangGraph com alertas
    critical = assistant.ask("Como conduzir paciente com INR elevado?", "PAC-002")
    check(
        "R9", "LangGraph emite alerta para resultado crítico",
        any(a["tipo"] == "resultado_critico" for a in critical["alertas"]),
        ["assistente_medico/assistant/graph.py (nó emitir_alertas)",
         "assistente_medico/db/repository.py (registrar_alerta)"],
        str(critical["alertas"]),
    )
    check(
        "R9", "Fluxo executa todas as etapas do grafo",
        len(critical["etapas"]) >= 7,
        ["assistente_medico/assistant/graph.py"],
        f"{len(critical['etapas'])} etapas",
    )

    # R10 — Projeto modular + README
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    check(
        "R10", "README documenta a Fase 3",
        "Fase 3" in readme and "main_fase3" in readme,
        ["README.md"],
    )
    check(
        "R10", "Projeto modularizado",
        (ROOT / "assistente_medico/assistant/graph.py").exists(),
        ["assistente_medico/ (pacote)", "main_fase3.py (CLI)"],
    )

    # R11 — Relatório técnico
    relatorio = ROOT / "docs/relatorio_tecnico_fase3.md"
    ok_rel = relatorio.exists()
    if ok_rel:
        texto = relatorio.read_text(encoding="utf-8")
        ok_rel = all(t in texto for t in ("fine-tuning", "LangGraph", "mermaid"))
    check(
        "R11", "Relatório técnico com diagrama e avaliação",
        ok_rel,
        ["docs/relatorio_tecnico_fase3.md",
         "docs/arquitetura_fase3.md",
         "outputs/fase3_evaluation.json (métricas base vs fine-tuned)"],
    )

    # R12 — Dataset anonimizado/sintético no repositório
    for name, path, extras in (
        ("Protocolos sintéticos", "data/hospital/protocolos.json",
         ["data/hospital/faq_medicos.json", "data/hospital/templates_documentos.json"]),
        ("Pacientes sintéticos", "data/patients/pacientes_sinteticos.json", []),
        ("Dataset de treino (JSONL)", "data/training/train.jsonl",
         ["data/training/valid.jsonl", "data/training/test.jsonl"]),
    ):
        check("R12", name, (ROOT / path).exists(), [path, *extras])

    repo.close()

    # ── Relatório final ────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("VALIDAÇÃO DOS CRITÉRIOS DE ACEITAÇÃO — FASE 3")
    print("=" * 78)
    failed = 0 if llm_ok else 1
    for req, desc, ok, arquivos, detail in RESULTS:
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        line = f"[{status}] {req}: {desc}"
        if detail and not ok:
            line += f"  -> {detail}"
        print(line)
        for arquivo in arquivos:
            print(f"       arquivo: {arquivo}")
    total = len(RESULTS) + 1  # +1 = check da LLM na seção inicial
    print("-" * 78)
    print(f"Resultado: {total - failed}/{total} verificações aprovadas (inclui o check da LLM)")
    if failed:
        print("ATENÇÃO: há critérios pendentes acima (FAIL).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
