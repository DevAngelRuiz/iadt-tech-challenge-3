# Notebooks

Notebook de demonstração da Fase 3: [`demo_fase3.ipynb`](./demo_fase3.ipynb) — dataset de fine-tuning, assistente LangGraph com paciente contextualizado, alertas automáticos, guardrails e auditoria.

Pré-requisitos (executar na raiz do repositório):

```bash
python main_fase3.py prepare
python main_fase3.py seed-db
```

O treino (`python main_fase3.py train`) é opcional para o notebook: sem o adapter LoRA, o assistente usa Ollama (se configurado no `.env`) ou o stub offline automaticamente.

Use o kernel do mesmo ambiente virtual em que instalou `requirements.txt`.
