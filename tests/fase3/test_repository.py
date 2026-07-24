import pytest

from assistente_medico.db.repository import HospitalRepository
from assistente_medico.db.seed_patients import SYNTHETIC_PATIENTS, seed_database


@pytest.fixture
def repo(tmp_path):
    repository = HospitalRepository(db_path=tmp_path / "test.db")
    yield repository
    repository.close()


def test_seed_e_busca_paciente(repo):
    seed_database(repo=repo, export_json=False)
    paciente = repo.get_paciente("PAC-001")
    assert paciente is not None
    assert paciente["alergias"] == "penicilina"
    assert len(repo.list_pacientes()) == len(SYNTHETIC_PATIENTS)


def test_exames_pendentes(repo):
    seed_database(repo=repo, export_json=False)
    pendentes = repo.exames_pendentes("PAC-001")
    nomes = {e["nome"] for e in pendentes}
    assert "biópsia por agulha grossa" in nomes
    assert all(e["status"] != "liberado" for e in pendentes)


def test_resultados_criticos(repo):
    seed_database(repo=repo, export_json=False)
    criticos = repo.resultados_criticos("PAC-002")
    assert len(criticos) == 1
    assert criticos[0]["nome"] == "INR"


def test_registrar_alerta(repo):
    seed_database(repo=repo, export_json=False)
    alerta_id = repo.registrar_alerta("PAC-002", "resultado_critico", "INR 5.8")
    alertas = repo.alertas_do_paciente("PAC-002")
    assert any(a["id"] == alerta_id for a in alertas)


def test_seed_idempotente(repo):
    seed_database(repo=repo, export_json=False)
    seed_database(repo=repo, export_json=False)
    assert len(repo.exames_do_paciente("PAC-001")) == 3
