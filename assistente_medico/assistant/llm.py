"""Wrapper da LLM do assistente (R1, R3).

Backends:
- mlx    → modelo fine-tuned com adapter LoRA local (produto do treino da Fase 3);
- ollama → LLM local via LangChain (`langchain-ollama`), fallback sem adapter;
- stub   → resposta determinística baseada no contexto recuperado (CI/offline).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from assistente_medico.config import resolve_assistant_llm_config

logger = logging.getLogger(__name__)


class AssistantLLM:
    """Interface única: generate(system, user) -> texto."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or resolve_assistant_llm_config()
        self.backend = self.config["backend"]
        self._mlx_model = None
        self._mlx_tokenizer = None
        self._ollama_chat = None

    # ── API pública ────────────────────────────────────────────────────

    def generate(self, system: str, user: str) -> str:
        if self.backend == "mlx":
            try:
                return self._generate_mlx(system, user)
            except Exception as exc:
                logger.warning("Backend MLX falhou (%s); usando stub determinístico.", exc)
                return self._generate_stub(user)
        if self.backend == "ollama":
            try:
                return self._generate_ollama(system, user)
            except Exception as exc:
                logger.warning("Backend Ollama falhou (%s); usando stub determinístico.", exc)
                return self._generate_stub(user)
        return self._generate_stub(user)

    # ── MLX (modelo fine-tuned) ────────────────────────────────────────

    def _generate_mlx(self, system: str, user: str) -> str:
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        if self._mlx_model is None:
            adapter = self.config.get("adapter_path")
            logger.info(
                "Carregando modelo MLX %s (adapter=%s)", self.config["model"], adapter
            )
            self._mlx_model, self._mlx_tokenizer = load(
                self.config["model"], adapter_path=adapter
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self._mlx_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return generate(
            self._mlx_model,
            self._mlx_tokenizer,
            prompt=prompt,
            max_tokens=400,
            sampler=make_sampler(temp=float(self.config.get("temperature", 0.1))),
            # Modelos 1B tendem a repetir parágrafos; penalidade evita loops.
            logits_processors=make_logits_processors(repetition_penalty=1.3),
            verbose=False,
        ).strip()

    # ── Ollama via LangChain ───────────────────────────────────────────

    def _generate_ollama(self, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_ollama import ChatOllama

        if self._ollama_chat is None:
            self._ollama_chat = ChatOllama(
                model=self.config["model"],
                base_url=self.config["base_url"],
                temperature=self.config["temperature"],
            )
        result = self._ollama_chat.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        return str(result.content).strip()

    # ── Stub determinístico (offline/CI) ──────────────────────────────

    def _generate_stub(self, user: str) -> str:
        """Compõe resposta a partir do contexto do prompt, sem LLM real.

        Extrai a seção de protocolos/literatura do prompt e devolve uma
        síntese segura com fontes — o suficiente para demonstrar o fluxo,
        os guardrails e a auditoria sem depender de rede.
        """
        protocols_section = _extract_section(user, "PROTOCOLOS E LITERATURA RELEVANTES:")
        exams_section = _extract_section(user, "EXAMES PENDENTES / NÃO LIBERADOS:")
        sources = re.findall(r"\(Fonte: ([^)]+)\)", protocols_section)
        fonte_txt = "; ".join(dict.fromkeys(sources)) if sources else "protocolos internos"

        parts = ["[resposta gerada em modo offline/stub]"]
        if protocols_section.strip():
            parts.append(
                "Com base nos protocolos e na literatura recuperados: "
                + protocols_section.strip()[:600]
            )
        if exams_section.strip() and "nenhum" not in exams_section.lower():
            parts.append(
                "Atenção aos exames pendentes listados no prontuário; recomenda-se "
                "verificar a liberação dos resultados antes de definir conduta."
            )
        parts.append(
            "Sugiro discutir o caso com o médico responsável: nenhuma conduta deve "
            f"ser adotada sem validação humana. (Fonte: {fonte_txt})"
        )
        return "\n\n".join(parts)


def _extract_section(text: str, header: str) -> str:
    if header not in text:
        return ""
    after = text.split(header, 1)[1]
    # Seção termina no próximo cabeçalho em caixa alta ou no fim do prompt.
    match = re.search(r"\n[A-ZÀ-Ú][A-ZÀ-Ú /()]+:\n", after)
    return after[: match.start()] if match else after
