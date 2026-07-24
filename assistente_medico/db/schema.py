"""Schema SQLite da base estruturada de prontuários (dados 100% sintéticos)."""

from __future__ import annotations

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS pacientes (
    id TEXT PRIMARY KEY,              -- identificador sintético (ex.: PAC-001)
    sexo TEXT NOT NULL,
    idade INTEGER NOT NULL,
    alergias TEXT NOT NULL DEFAULT '',            -- lista separada por ';'
    comorbidades TEXT NOT NULL DEFAULT '',
    medicamentos_em_uso TEXT NOT NULL DEFAULT '',
    hipotese_diagnostica TEXT NOT NULL DEFAULT '',
    setor TEXT NOT NULL DEFAULT 'enfermaria',
    atualizado_em TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS exames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id TEXT NOT NULL REFERENCES pacientes(id),
    nome TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pendente', 'coletado', 'liberado')),
    resultado TEXT,
    critico INTEGER NOT NULL DEFAULT 0,           -- 1 = valor crítico
    solicitado_em TEXT NOT NULL,
    liberado_em TEXT
);

CREATE TABLE IF NOT EXISTS prescricoes_pendentes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id TEXT NOT NULL REFERENCES pacientes(id),
    descricao TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'aguardando_validacao'
        CHECK (status IN ('aguardando_validacao', 'validada', 'rejeitada')),
    criado_em TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alertas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id TEXT NOT NULL REFERENCES pacientes(id),
    tipo TEXT NOT NULL,
    mensagem TEXT NOT NULL,
    origem TEXT NOT NULL DEFAULT 'assistente_medico',
    criado_em TEXT NOT NULL
);
"""
