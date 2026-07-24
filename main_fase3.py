"""Tech Challenge Fase 3 — CLI do assistente médico.

Comandos:
    python main_fase3.py prepare   # baixa PubMedQA + gera dados sintéticos + anonimiza
    python main_fase3.py seed-db   # popula o SQLite de prontuários sintéticos
    python main_fase3.py train     # fine-tuning LoRA (MLX, Apple Silicon)
    python main_fase3.py evaluate  # avalia modelo base vs. fine-tuned
    python main_fase3.py demo      # roda os fluxos de demonstração completos
    python main_fase3.py ask --pergunta "..." [--paciente PAC-001]  # pergunta única
"""

from __future__ import annotations

import argparse
import json
import logging

from assistente_medico.config import load_project_env

load_project_env()


def cmd_prepare(args: argparse.Namespace) -> None:
    from assistente_medico.data.prepare_dataset import prepare_dataset

    stats = prepare_dataset(max_pubmedqa=args.max_pubmedqa)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def cmd_seed_db(_: argparse.Namespace) -> None:
    from assistente_medico.db.seed_patients import seed_database

    print(json.dumps(seed_database(), ensure_ascii=False, indent=2))


def cmd_train(args: argparse.Namespace) -> None:
    from assistente_medico.fine_tuning.train_lora import train_lora

    metadata = train_lora(iters=args.iters, batch_size=args.batch_size)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def cmd_evaluate(args: argparse.Namespace) -> None:
    from assistente_medico.fine_tuning.evaluate import evaluate_models

    results = evaluate_models(max_examples=args.max_examples)
    print(json.dumps(results, ensure_ascii=False, indent=2))


def cmd_demo(_: argparse.Namespace) -> None:
    from assistente_medico.pipeline import print_demo_results, run_demo

    print_demo_results(run_demo())
    print("\nLog de auditoria: outputs/fase3_audit_log.jsonl")


def cmd_ask(args: argparse.Namespace) -> None:
    from assistente_medico.pipeline import build_assistant

    assistant = build_assistant()
    result = assistant.ask(pergunta=args.pergunta, paciente_id=args.paciente)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="Tech Challenge Fase 3 — Assistente Médico")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="Preparar dataset (PubMedQA + sintéticos)")
    p_prepare.add_argument("--max-pubmedqa", type=int, default=800)
    p_prepare.set_defaults(func=cmd_prepare)

    p_seed = sub.add_parser("seed-db", help="Popular banco de prontuários sintéticos")
    p_seed.set_defaults(func=cmd_seed_db)

    p_train = sub.add_parser("train", help="Fine-tuning LoRA (MLX)")
    p_train.add_argument("--iters", type=int, default=300)
    p_train.add_argument("--batch-size", type=int, default=2)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Avaliar base vs. fine-tuned")
    p_eval.add_argument("--max-examples", type=int, default=30)
    p_eval.set_defaults(func=cmd_evaluate)

    p_demo = sub.add_parser("demo", help="Rodar fluxos de demonstração")
    p_demo.set_defaults(func=cmd_demo)

    p_ask = sub.add_parser("ask", help="Fazer uma pergunta ao assistente")
    p_ask.add_argument("--pergunta", required=True)
    p_ask.add_argument("--paciente", default="")
    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
