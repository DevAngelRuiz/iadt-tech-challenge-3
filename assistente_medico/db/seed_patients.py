"""Seed de prontuários sintéticos (R12): pacientes fictícios com exames e pendências."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from assistente_medico.config import PATIENTS_DATA_DIR
from assistente_medico.db.repository import HospitalRepository

logger = logging.getLogger(__name__)

# Dados 100% sintéticos: sem nomes, CPFs ou qualquer identificador real.
SYNTHETIC_PATIENTS = [
    {
        "id": "PAC-001",
        "sexo": "F",
        "idade": 58,
        "alergias": "penicilina",
        "comorbidades": "hipertensão; diabetes tipo 2",
        "medicamentos_em_uso": "metformina; losartana",
        "hipotese_diagnostica": "nódulo mamário em investigação (BI-RADS 4)",
        "setor": "ambulatorio",
        "exames": [
            {"nome": "mamografia", "status": "liberado", "resultado": "BI-RADS 4 — nódulo espiculado QSE mama direita", "critico": False},
            {"nome": "biópsia por agulha grossa", "status": "pendente", "resultado": None, "critico": False},
            {"nome": "glicemia de jejum", "status": "liberado", "resultado": "142 mg/dL", "critico": False},
        ],
        "prescricoes_pendentes": ["Solicitação de biópsia aguardando validação do mastologista"],
    },
    {
        "id": "PAC-002",
        "sexo": "M",
        "idade": 67,
        "alergias": "",
        "comorbidades": "insuficiência cardíaca; fibrilação atrial",
        "medicamentos_em_uso": "varfarina; carvedilol; furosemida",
        "hipotese_diagnostica": "descompensação de IC em investigação",
        "setor": "enfermaria",
        "exames": [
            {"nome": "INR", "status": "liberado", "resultado": "5.8", "critico": True},
            {"nome": "BNP", "status": "coletado", "resultado": None, "critico": False},
            {"nome": "radiografia de tórax", "status": "pendente", "resultado": None, "critico": False},
        ],
        "prescricoes_pendentes": ["Ajuste de anticoagulação aguardando avaliação médica"],
    },
    {
        "id": "PAC-003",
        "sexo": "F",
        "idade": 74,
        "alergias": "dipirona",
        "comorbidades": "DPOC",
        "medicamentos_em_uso": "formoterol; budesonida",
        "hipotese_diagnostica": "pneumonia adquirida na comunidade",
        "setor": "emergencia",
        "exames": [
            {"nome": "lactato", "status": "liberado", "resultado": "4.2 mmol/L", "critico": True},
            {"nome": "hemocultura (2 amostras)", "status": "coletado", "resultado": None, "critico": False},
            {"nome": "gasometria arterial", "status": "pendente", "resultado": None, "critico": False},
        ],
        "prescricoes_pendentes": ["Esquema antibiótico empírico aguardando validação do plantonista"],
    },
    {
        "id": "PAC-004",
        "sexo": "M",
        "idade": 45,
        "alergias": "",
        "comorbidades": "",
        "medicamentos_em_uso": "",
        "hipotese_diagnostica": "dor torácica atípica em observação",
        "setor": "emergencia",
        "exames": [
            {"nome": "ECG 12 derivações", "status": "liberado", "resultado": "sem supra de ST", "critico": False},
            {"nome": "troponina (3h)", "status": "pendente", "resultado": None, "critico": False},
        ],
        "prescricoes_pendentes": [],
    },
    {
        "id": "PAC-005",
        "sexo": "F",
        "idade": 62,
        "alergias": "contraste iodado",
        "comorbidades": "doença renal crônica (TFG 38)",
        "medicamentos_em_uso": "metformina; enalapril",
        "hipotese_diagnostica": "avaliação pré-operatória de colecistectomia",
        "setor": "ambulatorio",
        "exames": [
            {"nome": "creatinina", "status": "liberado", "resultado": "1.8 mg/dL", "critico": False},
            {"nome": "tomografia de abdome com contraste", "status": "pendente", "resultado": None, "critico": False},
        ],
        "prescricoes_pendentes": ["Preparo renal pré-contraste aguardando validação"],
    },
    {
        "id": "PAC-006",
        "sexo": "M",
        "idade": 71,
        "alergias": "",
        "comorbidades": "hipertensão; tabagismo",
        "medicamentos_em_uso": "anlodipino",
        "hipotese_diagnostica": "suspeita de AVC isquêmico — início dos sintomas há 2h",
        "setor": "emergencia",
        "exames": [
            {"nome": "TC de crânio sem contraste", "status": "pendente", "resultado": None, "critico": False},
            {"nome": "glicemia capilar", "status": "liberado", "resultado": "98 mg/dL", "critico": False},
        ],
        "prescricoes_pendentes": [],
    },
]


def seed_database(repo: HospitalRepository | None = None, export_json: bool = True) -> dict:
    """Popula o SQLite com os pacientes sintéticos e exporta JSON versionável."""
    owns_repo = repo is None
    repo = repo or HospitalRepository()
    try:
        for patient in SYNTHETIC_PATIENTS:
            record = {k: v for k, v in patient.items() if k not in ("exames", "prescricoes_pendentes")}
            repo.upsert_paciente(record)
            existing = {e["nome"] for e in repo.exames_do_paciente(patient["id"])}
            for exame in patient["exames"]:
                if exame["nome"] not in existing:
                    repo.add_exame(
                        patient["id"],
                        exame["nome"],
                        status=exame["status"],
                        resultado=exame["resultado"],
                        critico=exame["critico"],
                    )
            existing_presc = {p["descricao"] for p in repo.prescricoes_pendentes(patient["id"])}
            for desc in patient["prescricoes_pendentes"]:
                if desc not in existing_presc:
                    repo.add_prescricao_pendente(patient["id"], desc)

        if export_json:
            PATIENTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
            out = PATIENTS_DATA_DIR / "pacientes_sinteticos.json"
            out.write_text(
                json.dumps(SYNTHETIC_PATIENTS, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("Pacientes sintéticos exportados para %s", out)

        return {"pacientes": len(SYNTHETIC_PATIENTS), "db_path": str(repo.db_path)}
    finally:
        if owns_repo:
            repo.close()
