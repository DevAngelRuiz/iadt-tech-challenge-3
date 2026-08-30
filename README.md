# Tech Challenge — Fase 3 — Assistente Médico com Fine-tuning + LangChain/LangGraph

Assistente virtual médico treinado com dados do hospital (sintéticos + PubMedQA), capaz de:

- Responder dúvidas de médicos com base em **protocolos internos** e **literatura médica**, citando as fontes;
- Consultar **base estruturada de prontuários** (SQLite) e contextualizar respostas com dados atualizados do paciente;
- Executar **fluxos de decisão automatizados** com **LangGraph** (verificar exames pendentes, sugerir próximos passos, emitir alertas para a equipe);
- Respeitar **limites de atuação**: nunca prescreve, nunca dá diagnóstico definitivo, sempre exige validação humana;
- Registrar **auditoria completa** (JSONL) e **explainability** (fontes por resposta).

## Requisitos

- Python 3.11+
- ~4 GB de RAM livres para treino/inferência do modelo 1B quantizado
- (Opcional) [Ollama](https://ollama.com) local como backend alternativo da LLM

### Compatibilidade por plataforma

| Funcionalidade | Windows/Linux | Mac Apple Silicon |
|---|:---:|:---:|
| Preparação do dataset, SQLite, testes e LangGraph | ✅ | ✅ |
| Inferência com Ollama ou stub | ✅ | ✅ |
| Fine-tuning e avaliação com MLX-LM | ❌ | ✅ |
| Inferência com o adapter LoRA original | ❌ | ✅ |

O fine-tuning foi projetado para **Mac Apple Silicon** porque usa MLX-LM, otimizado
para a memória unificada dos processadores Apple. Em Windows/Linux, todo o fluxo do
assistente pode ser executado com Ollama ou stub, mas esses backends **não utilizam
o adapter LoRA treinado pelo pipeline MLX**.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env   # opcional: configura backend da LLM
```

No Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Para usar um modelo Ollama já instalado no Windows/Linux:

```env
FASE3_LLM_BACKEND=ollama
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_TIMEOUT=180
LLM_TEMPERATURE=0.1
```

> **Importante:** Ollama é um backend alternativo para demonstrar o assistente e
> os fluxos LangGraph. Ele não comprova nem substitui o fine-tuning LoRA feito com MLX.

## Passo a passo (pipeline completo)

```bash
# 1. Preparar dataset: baixa PubMedQA + gera dados sintéticos do hospital,
#    anonimiza e cria data/training/{train,valid,test}.jsonl
python main_fase3.py prepare

# 2. Popular a base de prontuários sintéticos (outputs/hospital.db)
python main_fase3.py seed-db

# 3. Fine-tuning LoRA (somente Mac Apple Silicon) — ~8 min em M-series 16 GB
python main_fase3.py train --iters 300

# 4. Avaliar modelo base vs. fine-tuned (somente Mac Apple Silicon)
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
| `ollama` | `OLLAMA_MODEL` configurado e sem adapter | LLM local genérica via LangChain (`langchain-ollama`); não usa o fine-tuning MLX |
| `stub` | nenhum backend real disponível | Resposta determinística baseada no contexto recuperado (CI/demo offline) |

## Fine-tuning e evidências de execução

O produto do fine-tuning é o adapter `models/fase3_lora/adapters.safetensors`.
Após executar `train` e `evaluate` no Mac, o pipeline também gera:

| Artefato | Finalidade | Política sugerida |
|---|---|---|
| `models/fase3_lora/adapters.safetensors` | Pesos LoRA aprendidos | Link externo, GitHub Release, Hugging Face ou Git LFS |
| `models/fase3_lora/train_metadata.json` | Modelo base, hiperparâmetros e duração | Versionar no repositório |
| `outputs/fase3_train_log.txt` | Comando, training loss e validation loss | Versionar no repositório |
| `outputs/fase3_evaluation.json` | Métricas do modelo base e fine-tuned | Versionar no repositório |

Para comprovar que o assistente está usando a LLM customizada, a execução deve mostrar:

```text
LLM MLX local | modelo=mlx-community/Llama-3.2-1B-Instruct-4bit
adapter=models/fase3_lora
gerar_resposta: backend=mlx
```

O treinamento e a avaliação devem ser refeitos sempre que o dataset versionado for
alterado. As métricas publicadas no relatório devem corresponder ao mesmo commit do
dataset usado para gerar o adapter.

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
| Avaliação do modelo | `outputs/fase3_evaluation.json` (gerado no Mac) + relatório técnico |
