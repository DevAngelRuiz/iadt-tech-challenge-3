from assistente_medico.data.anonymize import anonymize_record, anonymize_text, contains_pii


def test_remove_cpf():
    result = anonymize_text("Paciente com CPF 123.456.789-01 internado.")
    assert "123.456.789-01" not in result.text
    assert "[CPF_REMOVIDO]" in result.text
    assert result.replacements["CPF"] == 1


def test_remove_nome_com_pronome():
    result = anonymize_text("A paciente Maria da Silva apresentou melhora.")
    assert "Maria da Silva" not in result.text
    assert "[NOME_REMOVIDO]" in result.text


def test_remove_email_e_telefone():
    result = anonymize_text("Contato: fulano@example.com ou (11) 98765-4321.")
    assert "fulano@example.com" not in result.text
    assert "98765-4321" not in result.text


def test_remove_telefone_sem_formatacao_quando_rotulado():
    result = anonymize_text("Telefone: 98765432")
    assert result.text == "Telefone: [TELEFONE_REMOVIDO]"


def test_preserva_pubmed_id_com_oito_digitos():
    original = "Evidência clínica (Fonte: PubMedQA pubid=18540901)"
    result = anonymize_text(original)
    assert result.text == original
    assert result.total_replacements == 0


def test_preserva_intervalo_de_anos():
    original = "Pacientes acompanhados durante o período 2000-2005."
    result = anonymize_text(original)
    assert result.text == original
    assert result.total_replacements == 0


def test_remove_cep_formatado_ou_rotulado():
    formatado = anonymize_text("Endereço com CEP 01310-930")
    sem_hifen = anonymize_text("CEP: 01310930")
    assert "01310-930" not in formatado.text
    assert sem_hifen.text == "CEP: [CEP_REMOVIDO]"


def test_remove_data_nascimento():
    result = anonymize_text("Nascido em 12/03/1960, evolui bem.")
    assert "12/03/1960" not in result.text


def test_texto_limpo_nao_altera():
    original = "Protocolo de sepse orienta lactato e hemoculturas antes do antibiótico."
    result = anonymize_text(original)
    assert result.text == original
    assert result.total_replacements == 0


def test_contains_pii():
    assert contains_pii("CPF 111.222.333-44")
    assert not contains_pii("recall e especificidade do modelo")


def test_anonymize_record():
    record = {"instruction": "Dr. João Souza pediu exame", "output": "ok", "origem": "x"}
    cleaned, total = anonymize_record(record, ("instruction", "output"))
    assert "João Souza" not in cleaned["instruction"]
    assert total >= 1
