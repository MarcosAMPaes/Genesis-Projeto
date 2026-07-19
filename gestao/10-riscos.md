# 10 — Matriz de Riscos

Escalas: Probabilidade (P) e Impacto (I) em Baixo / Médio / Alto. Revisar mensalmente; risco materializado vira issue com dono e prazo.

## Riscos técnicos

| # | Risco | P | I | Mitigação | Gatilho de alerta |
|---|---|---|---|---|---|
| T1 | Erro dimensional em campo acima da meta (calibração degrada fora da bancada ideal: iluminação, altura, vibração) | Alta | Alto | Perfil de calibração por sessão; objeto de referência em **toda** captura; LiDAR como checagem; log de erro contínuo desde o 1º dia | Erro médio > 2 mm em qualquer sessão |
| T2 | SAM falha em rochas escuras, reflexivas ou com veios que se confundem com o fundo | Média | Alto | Iluminação difusa controlada (LED 3000–6500K); fundo de alto contraste; conjunto de validação com casos difíceis; fallback com prompt manual na UI | IoU < 0,95 em alguma classe de rocha |
| T3 | Notebook Apple Silicon sem CUDA — SAM lento localmente | Alta | Médio | MPS para unidades; lotes na AWS g5.xlarge (120 h orçadas); avaliar MobileSAM/FastSAM | > 30 s/fragmento local |
| T4 | Estouro do orçamento AWS (R$ 2.200/ano) | Média | Médio | Billing alert mensal; processar em lote (não sob demanda); S3 lifecycle; compressão de máscaras | Gasto acumulado > 1/12 × mês decorrido |
| T5 | NFP falha em polígonos degenerados (côncavos, quase-colineares) → sobreposições ou crash | Média | Alto | Validador de sobreposição independente do NFP; suíte de casos patológicos; ε do Douglas-Peucker calibrado | Qualquer sobreposição em solução aceita |
| T6 | SA lento ou preso em ótimo local em painéis grandes | Média | Médio | Benchmark versionado; ajustar T0/vizinhança; BLF como fallback aceitável; paralelizar múltiplas seeds na GPU | > 10 min no benchmark padrão |
| T7 | Gabarito digital não "fecha" na montagem física (junta, tolerância de corte) | Média | Alto | Parâmetro de junta desde o Módulo D; montagem-teste interna antes do 1º piloto; feedback dos marmoristas no formato | Desvio visível no 1º painel físico |

## Riscos administrativos (FAPES)

| # | Risco | P | I | Mitigação | Gatilho |
|---|---|---|---|---|---|
| A1 | Âncora de datas errada (publicação real ≠ 15/04/2026) → prazos calculados errados | Média | Alto | **Confirmar a data no TO/SIGFAPES agora**; recalcular calendário na confirmação | Divergência encontrada |
| A2 | Perder o prazo da PC parcial (13/11/2026) | Baixa | Alto | Checklist 3 do doc 09 iniciado em 15/10; dossiês mensais prontos tornam o relatório compilação, não redação | 01/11 sem rascunho pronto |
| A3 | Não atingir 60% da 1ª parcela até o M5 | Média | Médio | Acompanhar execução na planilha do doc 04; equipamentos (R$ 23.400) já superam o gatilho — conferir NFs | 14/09 com gasto < R$ 10.800 |
| A4 | Despesa glosada (fora de rubrica, fora da conta, NF irregular) | Média | Alto | Regras do doc 04; conferência mensal; remanejamento sempre com autorização prévia | Qualquer NF fora do padrão |
| A5 | Certidão negativa vencida na hora da 2ª parcela | Média | Médio | Renovar CNDs em novembro/2026 (validades cobrindo dezembro) | CND com validade < 30 dias no M8 |
| A6 | Terceirizado da interface (R$ 1.500) atrasa ou entrega abaixo | Média | Médio | Especificação funcional pronta antes de contratar; marcos de entrega no contrato; integração via CLI já testável sem UI | Sem contrato assinado até 14/10/2026 |
| A7 | Entrega contratada do cronograma não concluída na vigência (improrrogável) | Baixa | Alto | Calendário operacional antecipado ~1 mês vs. contrato; replanejamento formal via 13.4 se necessário | Qualquer entrega > 30 dias atrasada vs. operacional |

## Riscos de negócio

| # | Risco | P | I | Mitigação | Gatilho |
|---|---|---|---|---|---|
| N1 | Pilotos não se concretizam (empresas sem tempo/interesse) | Média | Alto | Começar prospecção no M6 (antes do software final); usar Sindirochas/CETEMAG como porta de entrada; painéis-demonstração próprios como isca | < 2 pilotos confirmados até 14/12/2026 |
| N2 | Valor percebido baixo (mosaico automatizado não precifica acima do custo) | Média | Alto | Medir custo/m² real nos pilotos vs. artesanal; focar segmento premium/arquitetura; usar métricas de sustentabilidade como diferencial | Feedback de preço negativo em 2 pilotos |
| N3 | Cópia da abordagem por player maior | Baixa | Médio | Registro de software INPI (M10–M11); velocidade de mercado; relacionamento local como barreira | Concorrente anuncia solução similar |
| N4 | Dependência de uma pessoa (equipe de 3, coordenador concentra o técnico) | Média | Médio | Documentação viva (este /docs); código no GitHub; conhecimento compartilhado com equipe | Bus factor evidente em módulo crítico |

## Rotina de gestão de riscos

1. Revisão mensal (junto do Checklist 1 do doc 09): reavaliar P×I, checar gatilhos.
2. Gatilho disparado → abrir issue com dono, plano e prazo; registrar no dossiê do mês.
3. Riscos novos entram na matriz com data; riscos encerrados ficam registrados com desfecho.
