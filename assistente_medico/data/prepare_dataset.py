"""Pipeline de preparação do dataset de fine-tuning (R1, R2, R12).

Combina:
- PubMedQA (`qiaojin/PubMedQA`, config `pqa_labeled`) — Q&A clínico com base em
  literatura médica (dataset sugerido no enunciado);
- Dados sintéticos internos do hospital (protocolos, FAQ, templates, exemplos
  de segurança).

Etapas: download → curadoria → anonimização → formato chat → split
train/valid/test → JSONL em data/training/ (formato aceito pelo mlx-lm).
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from assistente_medico.config import (
    RAW_DATA_DIR,
    ROOT,
    TRAINING_DATA_DIR,
    ensure_dirs,
)
from assistente_medico.data.anonymize import anonymize_text
from assistente_medico.data.synthetic_hospital import (
    SAFETY_EXAMPLES,
    build_hospital_training_examples,
    export_hospital_documents,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Você é um assistente médico do hospital. Responda com base em protocolos "
    "internos e literatura médica, sempre citando a fonte. Nunca prescreva "
    "medicamentos, nunca dê diagnóstico definitivo e lembre que toda conduta "
    "exige validação de um profissional de saúde."
)

PUBMEDQA_DATASET = "qiaojin/PubMedQA"
PUBMEDQA_CONFIG = "pqa_labeled"

# Curadoria: exemplos de recusa segura são raros (4) frente aos ~800 de Q&A.
# Oversampling evita que o fine-tuning "desaprenda" os limites de atuação.
SAFETY_OVERSAMPLE = 10

DECISION_PT = {"yes": "Sim", "no": "Não", "maybe": "Possivelmente"}


def _load_pubmedqa(max_examples: int) -> list[dict[str, Any]]:
    """Baixa o PubMedQA (pqa_labeled) e converte em exemplos instruction/output."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("Biblioteca 'datasets' indisponível — seguindo só com dados sintéticos.")
        return []

    try:
        ds = load_dataset(PUBMEDQA_DATASET, PUBMEDQA_CONFIG, split="train")
    except Exception as exc:  # rede indisponível, HF fora do ar etc.
        logger.warning("Falha ao baixar PubMedQA (%s) — seguindo só com dados sintéticos.", exc)
        return []

    examples: list[dict[str, Any]] = []
    for row in ds:
        contexts = row.get("context", {}).get("contexts", [])
        context_text = " ".join(contexts)[:1500]
        decision = str(row.get("final_decision", "")).lower()
        long_answer = str(row.get("long_answer", "")).strip()
        if not long_answer or decision not in DECISION_PT:
            continue
        examples.append(
            {
                "instruction": str(row["question"]).strip(),
                "context": context_text,
                "output": (
                    f"{DECISION_PT[decision]}. {long_answer} "
                    f"(Fonte: PubMedQA pubid={row['pubid']})"
                ),
                "origem": f"pubmedqa:{row['pubid']}",
            }
        )
        if len(examples) >= max_examples:
            break
    logger.info("PubMedQA: %d exemplos curados.", len(examples))
    return examples


def _to_chat_record(example: dict[str, Any]) -> dict[str, Any]:
    """Converte para o formato chat (messages) aceito pelo mlx-lm."""
    user_content = example["instruction"]
    context = example.get("context", "")
    if context:
        user_content = f"{user_content}\n\nContexto (literatura):\n{context}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": anonymize_text(user_content).text},
            {"role": "assistant", "content": anonymize_text(example["output"]).text},
        ],
        "origem": example.get("origem", "desconhecida"),
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _path_for_stats(path: Path) -> str:
    """Usa caminho relativo ao projeto quando possível, evitando paths da máquina."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def prepare_dataset(
    max_pubmedqa: int = 800,
    seed: int = 42,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Gera train/valid/test JSONL. Retorna estatísticas do processo."""
    ensure_dirs()
    out_dir = output_dir or TRAINING_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dados internos sintéticos do hospital (sempre disponíveis, offline).
    hospital_paths = export_hospital_documents()
    hospital_examples = build_hospital_training_examples()
    # Oversampling dos exemplos de segurança (ver SAFETY_OVERSAMPLE).
    hospital_examples = hospital_examples + SAFETY_EXAMPLES * (SAFETY_OVERSAMPLE - 1)

    # 2. PubMedQA (dataset sugerido no enunciado).
    pubmed_examples = _load_pubmedqa(max_pubmedqa)

    # Cache bruto para auditoria da curadoria.
    raw_path = RAW_DATA_DIR / "curadoria_bruta.jsonl"
    _write_jsonl(raw_path, hospital_examples + pubmed_examples)

    # 3. Anonimização + formato chat.
    records = [_to_chat_record(ex) for ex in hospital_examples + pubmed_examples]

    # 4. Split reprodutível: exemplos de hospital são poucos e críticos —
    #    mantidos integralmente no treino; PubMedQA é dividido 80/10/10.
    rng = random.Random(seed)
    hospital_records = records[: len(hospital_examples)]
    pubmed_records = records[len(hospital_examples):]
    rng.shuffle(pubmed_records)

    n = len(pubmed_records)
    n_valid = max(1, int(n * 0.1)) if n else 0
    n_test = max(1, int(n * 0.1)) if n else 0
    test = pubmed_records[:n_test]
    valid = pubmed_records[n_test : n_test + n_valid]
    train = hospital_records + pubmed_records[n_test + n_valid :]
    rng.shuffle(train)

    if not valid:  # modo offline: garante arquivos válidos para o mlx-lm
        valid = train[-4:]
    if not test:
        test = train[-4:]

    paths = {
        "train": out_dir / "train.jsonl",
        "valid": out_dir / "valid.jsonl",
        "test": out_dir / "test.jsonl",
    }
    _write_jsonl(paths["train"], train)
    _write_jsonl(paths["valid"], valid)
    _write_jsonl(paths["test"], test)

    stats = {
        "hospital_examples": len(hospital_examples),
        "pubmedqa_examples": len(pubmed_examples),
        "train": len(train),
        "valid": len(valid),
        "test": len(test),
        "paths": {k: _path_for_stats(v) for k, v in paths.items()},
        "hospital_docs": {k: _path_for_stats(v) for k, v in hospital_paths.items()},
        "raw_cache": _path_for_stats(raw_path),
    }
    stats_path = out_dir / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Dataset pronto: %s", stats)
    return stats
