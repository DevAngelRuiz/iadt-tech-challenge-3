"""Classificação de perguntas: meta e fora de escopo."""

from assistente_medico.assistant.prompts import (
    is_meta_question,
    is_out_of_scope_question,
)


def test_meta_quem_e_voce():
    assert is_meta_question("Quem é você? E o que pode fazer?")
    assert is_meta_question("O que você pode fazer?")
    assert not is_out_of_scope_question("Quem é você?")


def test_fora_de_escopo_data():
    assert is_out_of_scope_question("Que dia é hoje?")
    assert is_out_of_scope_question("Que horas são?")
    assert is_out_of_scope_question("Como está o tempo?")
    assert is_out_of_scope_question("Conte uma piada")


def test_perguntas_clinicas_nao_sao_fora_de_escopo():
    assert not is_out_of_scope_question("Qual o protocolo de sepse na primeira hora?")
    assert not is_out_of_scope_question("Qual o próximo passo para BI-RADS 4?")
    assert not is_out_of_scope_question("Como conduzir paciente com INR elevado?")
    assert not is_out_of_scope_question("Qual a janela para trombólise no AVC isquêmico?")
    assert not is_meta_question("Qual o protocolo de sepse?")
