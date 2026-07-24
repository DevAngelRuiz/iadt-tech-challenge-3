"""Fine-tuning LoRA via MLX-LM (Apple Silicon) — requisito R1.

Usa o CLI oficial `mlx_lm lora` sobre o dataset preparado em data/training/
(train.jsonl / valid.jsonl no formato chat). O adapter resultante fica em
models/fase3_lora/ e é carregado automaticamente pelo assistente.

Modelo base: mlx-community/Llama-3.2-1B-Instruct-4bit (LLaMA, como sugere o
enunciado; versão quantizada 4 bits para caber em 16 GB de RAM).
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from assistente_medico.config import (
    ADAPTER_DIR,
    DEFAULT_BASE_MODEL_MLX,
    OUTPUTS_DIR,
    TRAINING_DATA_DIR,
    ensure_dirs,
)

logger = logging.getLogger(__name__)


def train_lora(
    base_model: str = DEFAULT_BASE_MODEL_MLX,
    iters: int = 300,
    batch_size: int = 2,
    num_layers: int = 8,
    learning_rate: float = 1e-5,
    data_dir: Path | None = None,
    adapter_dir: Path | None = None,
) -> dict:
    """Roda o fine-tuning LoRA e retorna metadados do treino.

    Parâmetros conservadores por padrão,
    batch 2, LoRA nas últimas 8 camadas.
    """
    ensure_dirs()
    data = data_dir or TRAINING_DATA_DIR
    adapter = adapter_dir or ADAPTER_DIR
    adapter.mkdir(parents=True, exist_ok=True)

    train_file = data / "train.jsonl"
    valid_file = data / "valid.jsonl"
    if not train_file.exists() or not valid_file.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {data}. Rode antes: python main_fase3.py prepare"
        )

    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model", base_model,
        "--train",
        "--data", str(data),
        "--adapter-path", str(adapter),
        "--iters", str(iters),
        "--batch-size", str(batch_size),
        "--num-layers", str(num_layers),
        "--learning-rate", str(learning_rate),
        "--save-every", str(max(50, iters // 4)),
        "--steps-per-report", "10",
        "--steps-per-eval", str(max(50, iters // 4)),
        "--max-seq-length", "1024",
    ]
    logger.info("Iniciando fine-tuning LoRA: %s", " ".join(cmd))
    started = datetime.now(timezone.utc)
    result = subprocess.run(cmd, capture_output=True, text=True)
    finished = datetime.now(timezone.utc)

    log_txt = OUTPUTS_DIR / "fase3_train_log.txt"
    log_txt.write_text(
        f"CMD: {' '.join(cmd)}\n\n--- STDOUT ---\n{result.stdout}\n\n--- STDERR ---\n{result.stderr}",
        encoding="utf-8",
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Fine-tuning falhou (código {result.returncode}). Veja {log_txt}.\n"
            f"Últimas linhas: {result.stderr.strip().splitlines()[-5:] if result.stderr else 'sem stderr'}"
        )

    metadata = {
        "base_model": base_model,
        "adapter_path": str(adapter),
        "iters": iters,
        "batch_size": batch_size,
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "train_file": str(train_file),
        "valid_file": str(valid_file),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_s": (finished - started).total_seconds(),
        "train_log": str(log_txt),
    }
    meta_path = adapter / "train_metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Fine-tuning concluído em %.0fs. Adapter em %s", metadata["duration_s"], adapter)
    return metadata
