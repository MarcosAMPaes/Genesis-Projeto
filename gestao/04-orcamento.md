# 04 — Orçamento Aprovado e Execução Financeira

Orçamento conforme Anexo II contratado. Total orçado: **R$ 29.670,00** (folga de R$ 330,00 sobre a subvenção de R$ 30.000,00). Contrapartida de R$ 300,00 em conta específica, à parte.

## Materiais permanentes — R$ 23.400,00

| Item | Especificação aprovada | Qtd. | Valor |
|---|---|---|---|
| Notebook alto desempenho | Apple M3/M4, 16 GB RAM, 512 GB SSD, GPU integrada para visão computacional/ML | 1 | R$ 12.000,00 |
| iPhone 15 Pro (sensor LiDAR) | LiDAR para medição de distâncias, câmera 48 MP, A17 Pro — correspondência pixel-centímetro | 1 | R$ 8.500,00 |

> **Aquisição real: iPhone 17 Pro** (modelo superior equivalente em função — LiDAR + câmera 48 MP). O quadro acima preserva o texto do item **aprovado** no Anexo II. Na prestação de contas, anexar justificativa de equivalência funcional junto à NF; se o valor ou descrição da NF divergir materialmente do aprovado, consultar a FAPES previamente (ver [09-compliance-fapes.md](09-compliance-fapes.md) e contradição C6 em `/docs/10`).
| Bancada de captura automatizada | Estrutura metálica regulável com trilhos, base antivibração, iluminação integrada, fixação de amostras | 1 | R$ 1.700,00 |
| Sistema de iluminação LED controlada | Intensidade e temperatura de cor ajustáveis (3000K–6500K), eliminação de sombras/reflexos | 1 | R$ 1.200,00 |

## Materiais de consumo — R$ 370,00

| Item | Especificação | Qtd. | Valor |
|---|---|---|---|
| Kit de calibração e referência | Tabuleiros de xadrez em escalas variadas, objetos de referência certificados, réguas milimétricas | 1 | R$ 70,00 |
| Amostras de teste diversificadas | Fragmentos controlados de granito, mármore, quartzito e outras rochas | 1 | R$ 300,00 |

## Passagens, diárias e hospedagem — R$ 1.200,00

| Item | Especificação | Qtd. | Valor |
|---|---|---|---|
| Visitas técnicas regionais | Deslocamentos a empresas parceiras em Cachoeiro de Itapemirim e região (validação de campo, pilotos, feedback) — R$ 200/visita | 6 | R$ 1.200,00 |

## Serviços de terceiros — R$ 3.700,00

| Item | Especificação | Qtd. | Valor |
|---|---|---|---|
| AWS EC2 — instâncias GPU | g5.xlarge (A10G 24 GB) on-demand para picos de SAM/otimização; ~120 h/ano + EBS gp3 100 GB + buffer egress | 100 h | R$ 1.800,00 |
| AWS S3 — armazenamento | Standard, banco de fragmentos mapeados (~500 GB de imagens/máscaras/polígonos × 12 meses) | 500 GB | R$ 400,00 |
| Desenvolvimento de interface gráfica | UI/UX para aplicação industrial, por terceiros | 1 | R$ 1.500,00 |

## Outras despesas — R$ 1.000,00

| Item | Especificação | Valor |
|---|---|---|
| DOACI | Despesas Operacionais e Administrativas de Caráter Indivisível (Lei 10.973/04) | R$ 1.000,00 |

## Fluxo financeiro

| Evento | Valor | Quando |
|---|---|---|
| 1ª parcela | R$ 18.000,00 | Autorizada 27/04/2026, programada 30/04/2026 |
| Gatilho 60% | R$ 10.800,00 gastos/comprometidos | Conferir até 14/09/2026 (M5) |
| 2ª parcela | R$ 12.000,00 | ~14/12/2026, após requisitos do item 12.5 |
| Contrapartida | R$ 300,00 (total, em parcelas) | Conforme cronograma de desembolso; 2ª parte exigida para a 2ª parcela |

## Regras de execução (não negociáveis)

1. **Toda despesa** deve estar prevista no orçamento aprovado e paga pela conta Banestes específica; remanejamento entre rubricas exige autorização prévia da FAPES (item 13.4 + Resolução CCAF 309/2022 e 313/2022 + Manual de Utilização de Recursos).
2. **NF/recibo em nome da empresa**, com data **dentro da vigência** (15/04/2026–14/04/2027), arquivados para a prestação de contas.
3. Rendimentos de aplicação financeira devem ser aplicados no objeto do projeto (15.1.d).
4. Saldo não usado (incl. rendimentos) é devolvido em até 15 dias após o fim.
5. Manter planilha de conciliação: data, item do orçamento, fornecedor, NF, valor, parcela de origem.

## Controle de execução (preencher continuamente)

| Rubrica | Orçado | Executado | Saldo | NFs arquivadas? |
|---|---|---|---|---|
| Materiais permanentes | R$ 23.400,00 | — | — | — |
| Materiais de consumo | R$ 370,00 | — | — | — |
| Passagens/diárias | R$ 1.200,00 | — | — | — |
| Serviços de terceiros | R$ 3.700,00 | — | — | — |
| DOACI | R$ 1.000,00 | — | — | — |
| **Total** | **R$ 29.670,00** | — | — | — |

> Alerta AWS: os R$ 2.200 de nuvem cobrem ~120 h de GPU e 500 GB. Monitorar billing mensalmente; estouro de nuvem não tem cobertura no orçamento sem remanejamento autorizado.
