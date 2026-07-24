"""Configuração central da Fase 3: paths, variáveis de ambiente e limites de segurança."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = project_root()

DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
HOSPITAL_DATA_DIR = DATA_DIR / "hospital"
PATIENTS_DATA_DIR = DATA_DIR / "patients"
TRAINING_DATA_DIR = DATA_DIR / "training"

OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = ROOT / "models"
ADAPTER_DIR = MODELS_DIR / "fase3_lora"

DB_PATH = OUTPUTS_DIR / "hospital.db"
AUDIT_LOG_PATH = OUTPUTS_DIR / "fase3_audit_log.jsonl"

# ── Defaults de LLM ────────────────────────────────────────────────────────

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "phi3:mini"
DEFAULT_OLLAMA_TIMEOUT = 180
DEFAULT_LLM_TEMPERATURE = 0.1

# Modelo base para fine-tuning LoRA via MLX (não exige aprovação/gated access).
DEFAULT_BASE_MODEL_MLX = "mlx-community/Llama-3.2-1B-Instruct-4bit"

# ── Limites de atuação do assistente (requisito de segurança do desafio) ──

SAFETY_RULES = (
    "NUNCA prescrever medicamentos, posologias ou tratamentos de forma definitiva.",
    "Toda sugestão clínica exige validação humana por profissional habilitado.",
    "NUNCA fazer diagnóstico definitivo de paciente individual.",
    "NUNCA solicitar ou expor dados pessoais identificáveis (nome, CPF, endereço).",
    "Sempre indicar as fontes utilizadas na resposta (protocolo, literatura, prontuário).",
    "Em situação de emergência, orientar acionamento imediato da equipe médica.",
)

DISCLAIMER = (
    "AVISO: esta resposta é um apoio à decisão clínica gerado por IA. "
    "Ela NÃO substitui a avaliação de um profissional de saúde e requer validação humana."
)


def load_project_env(override: bool = True) -> Path | None:
    """Carrega variáveis do arquivo .env na raiz do projeto."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return None
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.warning("python-dotenv não instalado; .env ignorado.")
        return env_path
    load_dotenv(env_path, override=override)
    return env_path


def ensure_dirs() -> None:
    for d in (RAW_DATA_DIR, HOSPITAL_DATA_DIR, PATIENTS_DATA_DIR, TRAINING_DATA_DIR, OUTPUTS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _adapter_available() -> bool:
    return (ADAPTER_DIR / "adapters.safetensors").exists()


def resolve_assistant_llm_config() -> dict[str, Any]:
    """Resolve o backend da LLM do assistente.

    Prioridade:
    1. FASE3_LLM_BACKEND explícito (mlx | ollama | stub);
    2. Adapter LoRA fine-tuned disponível → mlx;
    3. OLLAMA_MODEL configurado → ollama;
    4. stub determinístico (CI / demo offline).
    """
    explicit = os.environ.get("FASE3_LLM_BACKEND", "").strip().lower()
    temperature = _get_float("LLM_TEMPERATURE", DEFAULT_LLM_TEMPERATURE)

    def _mlx_config() -> dict[str, Any]:
        return {
            "backend": "mlx",
            "model": os.environ.get("FASE3_BASE_MODEL", DEFAULT_BASE_MODEL_MLX).strip(),
            "adapter_path": str(ADAPTER_DIR) if _adapter_available() else None,
            "temperature": temperature,
        }

    def _ollama_config() -> dict[str, Any]:
        return {
            "backend": "ollama",
            "model": os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip() or DEFAULT_OLLAMA_MODEL,
            "base_url": os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).strip(),
            "timeout_s": _get_int("OLLAMA_TIMEOUT", DEFAULT_OLLAMA_TIMEOUT),
            "temperature": temperature,
        }

    def _stub_config() -> dict[str, Any]:
        return {"backend": "stub", "model": "offline_stub", "temperature": temperature}

    if explicit == "mlx":
        return _mlx_config()
    if explicit == "ollama":
        return _ollama_config()
    if explicit == "stub":
        return _stub_config()

    if _adapter_available():
        return _mlx_config()
    if os.environ.get("OLLAMA_MODEL", "").strip():
        return _ollama_config()
    return _stub_config()


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip())
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)).strip())
    except ValueError:
        return default


def describe_llm_config(cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or resolve_assistant_llm_config()
    backend = cfg["backend"]
    if backend == "mlx":
        adapter = cfg.get("adapter_path") or "sem adapter (modelo base)"
        return f"LLM MLX local | modelo={cfg['model']} | adapter={adapter}"
    if backend == "ollama":
        return f"LLM Ollama | modelo={cfg['model']} | url={cfg['base_url']}"
    return "LLM em modo stub offline (nenhum backend real configurado)"
