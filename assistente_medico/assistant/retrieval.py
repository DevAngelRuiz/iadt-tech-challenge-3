"""Recuperação de contexto (protocolos internos + literatura PubMedQA).

Retriever lexical simples (score por sobreposição de termos), suficiente para
contextualizar a LLM com fontes rastreáveis sem dependência de embeddings —
funciona 100% offline, requisito para demo e CI.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from assistente_medico.config import TRAINING_DATA_DIR
from assistente_medico.data.synthetic_hospital import FAQS, PROTOCOLS

_STOPWORDS = {
    "o", "a", "os", "as", "de", "do", "da", "dos", "das", "e", "em", "no", "na",
    "nos", "nas", "um", "uma", "para", "por", "com", "que", "qual", "quais",
    "quando", "como", "se", "ao", "à", "é", "ser", "the", "of", "in", "and",
    "is", "to", "for", "a", "an",
}


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", _normalize(text))
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


class KnowledgeBase:
    """Base de conhecimento consultável: protocolos, FAQ e literatura."""

    def __init__(self, training_dir: Path | None = None):
        self._docs: list[dict[str, Any]] = []
        for proto in PROTOCOLS:
            self._docs.append(
                {
                    "ref": f"protocolo:{proto['id']}",
                    "titulo": proto["titulo"],
                    "texto": proto["conteudo"],
                }
            )
        for faq in FAQS:
            self._docs.append(
                {
                    "ref": f"faq:{faq['fonte']}",
                    "titulo": faq["pergunta"],
                    "texto": faq["resposta"],
                }
            )
        self._load_pubmedqa(training_dir or TRAINING_DATA_DIR)
        self._doc_tokens = [_tokens(f"{d['titulo']} {d['texto']}") for d in self._docs]

    def _load_pubmedqa(self, training_dir: Path) -> None:
        """Indexa exemplos PubMedQA já curados/anonimizados do dataset de treino."""
        for name in ("train.jsonl", "valid.jsonl"):
            path = training_dir / name
            if not path.exists():
                continue
            # Iteração por arquivo divide apenas em '\n' (splitlines quebraria
            # em separadores Unicode como \u2028 presentes em textos do PubMed).
            with path.open(encoding="utf-8") as f:
                lines = list(f)
            for line in lines:
                if not line.strip():
                    continue
                rec = json.loads(line)
                origem = rec.get("origem", "")
                if not origem.startswith("pubmedqa:"):
                    continue
                messages = rec.get("messages", [])
                user = next((m["content"] for m in messages if m["role"] == "user"), "")
                assistant = next(
                    (m["content"] for m in messages if m["role"] == "assistant"), ""
                )
                self._docs.append(
                    {
                        "ref": origem,
                        "titulo": user.split("\n")[0][:160],
                        "texto": assistant[:800],
                    }
                )

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        scored = []
        for doc, doc_tokens in zip(self._docs, self._doc_tokens):
            overlap = len(query_tokens & doc_tokens)
            if overlap:
                scored.append((overlap / len(query_tokens), doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    @property
    def size(self) -> int:
        return len(self._docs)


def format_context_docs(docs: list[dict[str, Any]]) -> str:
    if not docs:
        return "Nenhum protocolo ou literatura diretamente relacionado foi encontrado."
    blocks = []
    for doc in docs:
        blocks.append(f"[{doc['ref']}] {doc['titulo']}\n{doc['texto']} (Fonte: {doc['ref']})")
    return "\n\n".join(blocks)
