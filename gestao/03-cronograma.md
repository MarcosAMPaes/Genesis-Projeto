# 03 — Cronograma

## Âncora de datas

Item 5.5 do edital: os 12 meses correm da **publicação do extrato do Termo de Outorga no DIO/ES**, prazo improrrogável. Como o pagamento da 1ª parcela foi autorizado em 27/04/2026 e programado para 30/04/2026 (a liberação só ocorre após a publicação), a publicação ocorreu em abril/2026. **Data adotada: 15/04/2026.** Se a data real (verificável no TO ou no SIGFAPES → Área do Projeto Contratado → item 1.2) for outra, deslocar todas as datas pela mesma diferença de dias.

## Cronograma físico contratado (Anexo II — referência para prestação de contas)

Este é o quadro contra o qual a FAPES cobra as entregas. Extraído fielmente do formulário contratado:

| Entrega | M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 | M9 | M10 | M11 | M12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Aquisição e configuração de equipamentos | X | | | | | | | | | | | |
| Bancada automatizada de captura | | X | X | | | | | | | | | |
| Sistema de calibração (método de Zhang) | | X | X | X | | | | | | | | |
| Otimização dos algoritmos SAM e Douglas-Peucker | | | | X | X | X | | | | | | |
| Algoritmos de empacotamento integrados | | | | | | X | X | | | | | |
| Interface de usuário e sistema de controle | | | | | | | | X | X | | | |
| Testes piloto com empresas parceiras locais | | | | | | | | | X | X | X | |
| Validação técnica e ajustes de performance | | | | | | | | | | | X | |
| Estratégia de comercialização e PI | | | | | | | | | | | X | X |
| Implementação comercial e licenciamento | | | | | | | | | | | | X |

## Calendário operacional (ancorado em 15/04/2026)

Plano de trabalho interno. Antecipa levemente algumas entregas em relação ao contratado (interface e pilotos ~1 mês antes), criando margem de segurança — o contrato é o teto, não o piso.

| Mês | Período | Entrega técnica prevista | Obrigação FAPES |
|---|---|---|---|
| M1 | 15/04–14/05/26 | Aquisição e configuração de equipamentos | Informar dados da conta Banestes (30 dias do início) |
| M2 | 15/05–14/06/26 | Bancada automatizada de captura | Executar e arquivar NFs |
| M3 | 15/06–14/07/26 | Bancada + calibração de Zhang | — |
| M4 | 15/07–14/08/26 | Calibração Zhang + SAM/Douglas-Peucker | **← Você está aqui** |
| M5 | 15/08–14/09/26 | Calibração Zhang + SAM/Douglas-Peucker | Checar: ≥ 60% da 1ª parcela gasto (R$ 10.800)? |
| M6 | 15/09–14/10/26 | SAM/Douglas-Peucker + empacotamento | Fim do 6º mês em 14/10/26 — abre prazo da PC parcial |
| M7 | 15/10–14/11/26 | Empacotamento + interface de usuário | **PC parcial (técnica + financeira) até 13/11/26** |
| M8 | 15/11–14/12/26 | Interface + início dos testes piloto | 2ª parcela (R$ 12.000) liberada ~14/12/26 |
| M9 | 15/12/26–14/01/27 | Testes piloto com empresas parceiras | Aporte da 2ª contrapartida + certidões negativas |
| M10 | 15/01–14/02/27 | Testes piloto, validação técnica, estratégia de PI | Executar a 2ª parcela |
| M11 | 15/02–14/03/27 | Estratégia de comercialização e PI | Últimas compras — NF deve caber na vigência |
| M12 | 15/03–14/04/27 | Implementação comercial e licenciamento | **Fim da vigência em 14/04/27** |
| Pós | até 29/04/27 | — | Devolver saldo remanescente (15 dias) |
| Pós | até 14/05/27 | — | PC final (técnica + financeira) |

## Datas-limite críticas (ordem cronológica)

| Data | O quê | Consequência se perder |
|---|---|---|
| 14/09/2026 | ≥ 60% da 1ª parcela comprometido/gasto | Atrasa a 2ª parcela |
| 13/11/2026 | Prestação de contas parcial no SIGFAPES | Inadimplência técnica → trava 2ª parcela, risco de suspensão |
| ~14/12/2026 | Requisitos completos da 2ª parcela (PC + contrapartida + CNDs) | Sem os R$ 12.000 para pilotos e fase final |
| 14/04/2027 | Fim da vigência — última data para despesas (NF dentro da vigência) | Despesa fora da vigência é glosada |
| 29/04/2027 | Devolução de saldo remanescente | Causa de cancelamento/ressarcimento (15.1.h) |
| 14/05/2027 | Prestação de contas final | Inadimplência → ressarcimento + suspensão de até 24 meses |

## Regras de manutenção deste cronograma

1. Confirmada a data real de publicação do TO, atualizar a âncora e recalcular a coluna "Período".
2. Qualquer replanejamento que altere entregas do quadro contratado exige solicitação prévia à FAPES (item 13.4).
3. Registrar evidências por mês (fotos da bancada, logs de calibração, commits, NFs) — insumo direto do relatório técnico.
