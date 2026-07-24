"""Demo da Fase 3 — roteiro para o vídeo de apresentação.

Executa, em sequência:
1. Estatísticas do dataset preparado (fine-tuning);
2. Configuração da LLM ativa (fine-tuned MLX, Ollama ou stub);
3. Fluxos automatizados do LangGraph com pacientes sintéticos
   (contextualização, alertas, guardrails);
4. Últimas entradas do log de auditoria.

Uso:
    .venv/bin/python scripts/demo_fase3.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistente_medico.config import (  # noqa: E402
    AUDIT_LOG_PATH,
    TRAINING_DATA_DIR,
    describe_llm_config,
    load_project_env,
)

logging.basicConfig(level=logging.WARNING)
load_project_env()

from assistente_medico.assistant.logging_audit import AuditLogger  # noqa: E402
from assistente_medico.pipeline import build_assistant, print_demo_results, run_demo  # noqa: E402


def main() -> None:
    print("=" * 78)
    print("TECH CHALLENGE FASE 3 — ASSISTENTE MÉDICO (demo)")
    print("=" * 78)

    stats_path = TRAINING_DATA_DIR / "dataset_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        print("\n[1] Dataset de fine-tuning:")
        print(f"    PubMedQA: {stats['pubmedqa_examples']} exemplos | "
              f"Hospital (sintético): {stats['hospital_examples']} exemplos")
        print(f"    Split: train={stats['train']} valid={stats['valid']} test={stats['test']}")
    else:
        print("\n[1] Dataset ainda não preparado — rode: python main_fase3.py prepare")

    assistant = build_assistant()
    print(f"\n[2] {describe_llm_config(assistant.llm.config)}")
    print(f"    Base de conhecimento: {assistant.kb.size} documentos indexados")

    print("\n[3] Fluxos automatizados (LangGraph):")
    results = run_demo(assistant)
    print_demo_results(results)

    print("\n[4] Auditoria (últimas 3 entradas):")
    audit = AuditLogger()
    for entry in audit.read_all()[-3:]:
        print(
            f"    {entry['timestamp']} | sessao={entry['session_id']} | "
            f"paciente={entry['paciente_id'] or '(nenhum)'} | "
            f"bloqueada={entry['bloqueada_pelo_guardrail']} | "
            f"fontes={len(entry['fontes'])} | etapas={len(entry['etapas'])}"
        )
    print(f"\nLog completo: {AUDIT_LOG_PATH}")


if __name__ == "__main__":
    main()
