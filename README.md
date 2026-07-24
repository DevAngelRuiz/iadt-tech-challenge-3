# Tech Challenge — Fase 3 — Assistente Médico com Fine-tuning + LangChain/LangGraph

Assistente virtual médico treinado com dados do hospital (sintéticos + PubMedQA), capaz de:

- Responder dúvidas de médicos com base em **protocolos internos** e **literatura médica**, citando as fontes;
- Consultar **base estruturada de prontuários** (SQLite) e contextualizar respostas com dados atualizados do paciente;
- Executar **fluxos de decisão automatizados** com **LangGraph** (verificar exames pendentes, sugerir próximos passos, emitir alertas para a equipe);
- Respeitar **limites de atuação**: nunca prescreve, nunca dá diagnóstico definitivo, sempre exige validação humana;
- Registrar **auditoria completa** (JSONL) e **explainability** (fontes por resposta).

## Requisitos

- Python 3.11+ (testado em Mac Apple Silicon — o fine-tuning usa MLX)
- ~4 GB de RAM livres para treino/inferência do modelo 1B quantizado
- (Opcional) [Ollama](https://ollama.com) local como backend alternativo da LLM

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env   # opcional: configura backend da LLM
```

## Passo a passo (pipeline completo)

```bash
# 1. Preparar dataset: baixa PubMedQA + gera dados sintéticos do hospital,
#    anonimiza e cria data/training/{train,valid,test}.jsonl
python main_fase3.py prepare

# 2. Popular a base de prontuários sintéticos (outputs/hospital.db)
python main_fase3.py seed-db

# 3. Fine-tuning LoRA (MLX) — ~8 min em um Mac M-series 16 GB
python main_fase3.py train --iters 300

# 4. Avaliar modelo base vs. fine-tuned (acurácia, citação de fontes, segurança)
python main_fase3.py evaluate

# 5. Demo completa dos fluxos LangGraph (usa o modelo fine-tuned automaticamente)
python main_fase3.py demo

# Pergunta avulsa com contexto de paciente:
python main_fase3.py ask --pergunta "Qual o próximo passo desta paciente?" --paciente PAC-001
```

Validação dos critérios de aceitação (R1–R12): `python scripts/validate_fase3.py`
— o script mostra qual LLM está ativa (fine-tuned MLX, Ollama ou stub) e, em cada
check, os arquivos do projeto que implementam aquele requisito (guia de leitura do código).

Testes: `pytest`

## Backends da LLM

O assistente resolve o backend automaticamente (configurável via `FASE3_LLM_BACKEND` no `.env`):

| Backend | Quando é usado | Descrição |
|---------|----------------|-----------|
| `mlx` | adapter em `models/fase3_lora/` existe | **Modelo fine-tuned** (Llama 3.2 1B 4-bit + LoRA) — produto da Fase 3 |
| `ollama` | `OLLAMA_MODEL` configurado e sem adapter | LLM local via LangChain (`langchain-ollama`) |
| `stub` | nenhum backend real disponível | Resposta determinística baseada no contexto recuperado (CI/demo offline) |

## Estrutura

```
assistente_medico/
  config.py                 # paths, env, limites de segurança
  data/                     # anonimização, sintéticos, preparação do dataset
  fine_tuning/              # treino LoRA (MLX) e avaliação before/after
  db/                       # SQLite: pacientes, exames, prescrições, alertas
  assistant/                # LLM, retrieval, tools, guardrails, grafo LangGraph
  pipeline.py               # cenários de demonstração
main_fase3.py               # CLI da Fase 3
data/hospital/              # protocolos, FAQ e templates sintéticos (versionados)
data/patients/              # prontuários sintéticos (versionados)
data/training/              # dataset de fine-tuning anonimizado (versionado)
docs/arquitetura_fase3.md   # diagrama e componentes
docs/relatorio_tecnico_fase3.md  # relatório técnico da entrega
outputs/                    # hospital.db, audit log, avaliação, logs de treino
models/fase3_lora/          # adapter LoRA (gerado pelo treino; não versionado)
```

## Dataset

- **PubMedQA** (`qiaojin/PubMedQA`, config `pqa_labeled`, ~1k Q&A revisados por especialistas) — sugestão do enunciado; convertido para chat PT-BR-friendly com decisão (Sim/Não/Possivelmente) + justificativa + `pubid` como fonte.
- **Dados internos sintéticos** do "Hospital Vida Plena (fictício)": 12 protocolos, 15 FAQs de médicos, 5 templates de laudo/receita/procedimento e 4 exemplos de recusa segura.
- Todo texto passa por **anonimização** (CPF, RG, telefone, e-mail, datas, nomes, nº de prontuário) antes de entrar no dataset — ver `assistente_medico/data/anonymize.py`.

## Segurança e auditoria

- Guardrail de **entrada**: pedidos de prescrição, diagnóstico definitivo ou dados pessoais são recusados antes da LLM.
- Guardrail de **saída**: respostas com posologia explícita são sanitizadas; disclaimer clínico obrigatório.
- **Auditoria**: cada interação gera entrada em `outputs/fase3_audit_log.jsonl` com prompt, resposta bruta, violações, fontes, alertas e todas as etapas do grafo.
- **Explainability**: fontes rastreáveis por resposta (`protocolo:PROTO-XXX`, `pubmedqa:<pubid>`, `prontuario:<id>/<campo>`).

## Entregáveis

| Item do enunciado | Onde está |
|---|---|
| Pipeline de fine-tuning | `assistente_medico/fine_tuning/` + `main_fase3.py train` |
| Integração com LangChain | `assistente_medico/assistant/` (tools, LLM, prompts) |
| Fluxos do LangGraph | `assistente_medico/assistant/graph.py` |
| Dataset anonimizado/sintético | `data/hospital/`, `data/patients/`, `data/training/` |
| Relatório técnico | [`docs/relatorio_tecnico_fase3.md`](./docs/relatorio_tecnico_fase3.md) |
| Diagrama do fluxo | [`docs/arquitetura_fase3.md`](./docs/arquitetura_fase3.md) |
| Avaliação do modelo | `outputs/fase3_evaluation.json` + relatório técnico |
