from assistente_medico.assistant.guardrails import check_request, check_response
from assistente_medico.config import DISCLAIMER


def test_bloqueia_pedido_de_prescricao():
    result = check_request("Prescreva amoxicilina 500mg de 8 em 8 horas.")
    assert not result.allowed
    assert "pedido_de_prescricao" in result.violations
    assert "validação humana" in result.final_text


def test_bloqueia_ajuste_de_dose():
    result = check_request("Aumente a dose do anticoagulante do paciente.")
    assert not result.allowed


def test_bloqueia_diagnostico_definitivo():
    result = check_request("Me dê o diagnóstico definitivo desse caso.")
    assert not result.allowed
    assert "pedido_de_diagnostico_definitivo" in result.violations


def test_bloqueia_pedido_de_pii():
    result = check_request("Qual o CPF do paciente do leito 12?")
    assert not result.allowed
    assert "pedido_de_dados_pessoais" in result.violations


def test_permite_pergunta_clinica_legitima():
    result = check_request("Qual o protocolo de sepse na primeira hora?")
    assert result.allowed


def test_resposta_com_posologia_e_sanitizada():
    raw = "Tome 500 mg de amoxicilina a cada 8 horas por 7 dias."
    result = check_response(raw)
    assert result.sanitized
    assert "posologia_explicita" in result.violations
    assert "500 mg" not in result.final_text.replace(DISCLAIMER, "")


def test_resposta_limpa_recebe_disclaimer():
    raw = "O protocolo orienta coleta de lactato. (Fonte: PROTO-001)"
    result = check_response(raw)
    assert not result.sanitized
    assert DISCLAIMER in result.final_text
    assert result.requer_validacao_humana
