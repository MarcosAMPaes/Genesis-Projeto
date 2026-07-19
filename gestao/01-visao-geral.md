# 01 — Visão Geral do Projeto

## Identificação

| Campo | Valor |
|---|---|
| Nome comercial | **Petra Smart** |
| Título oficial da proposta | Utilização de Visão Computacional e Algoritmos de Otimização para Criação de Revestimento de Pedra Natural a Partir de Retalhos de Rochas Ornamentais Naturais |
| Programa | Gênesis — Geração de Ideias Inovadoras e Estímulo à Abertura de Startups |
| Edital | FAPES nº 09/2025 (Central-Sul) — processo 2024-0RDMG |
| Classificação Etapa II | 9º lugar entre 20 contempladas — nota final 87,8/100 (resultado preliminar de 30/09/2025) |
| Coordenador | Marcos Antonio Mendes Paes |
| Equipe | Igor Henrique Beloti Pizetta; Dullye Noleto Lima Teixeira |
| Município | Cachoeiro de Itapemirim/ES |
| Subvenção | R$ 30.000,00 (FUNCITEC, fonte 0159) + contrapartida de R$ 300,00 (1%) |
| Vigência | 15/04/2026 → 14/04/2027 (12 meses, improrrogável) |

> **Observação documental:** o cabeçalho do formulário Anexo II submetido menciona "Edital FAPES nº 03/2025 — processo 2024-3BN5Z" (provável template de outra microrregião). O edital que rege o projeto é o **09/2025 (2024-0RDMG)**. Confirmar o número de processo correto no Termo de Outorga e usá-lo em toda correspondência (exigência do item 19.1 do edital).

## O problema

O setor de rochas ornamentais gera grandes volumes de fragmentos irregulares descartados nas etapas de corte, polimento e acabamento. Esses resíduos representam desperdício de material nobre e passivo ambiental. Quando reaproveitados, viram agregado para concreto, tijolo ecológico ou pavimentação — aplicações que destroem o valor estético do material e geram baixo valor agregado. A variabilidade geométrica dos fragmentos inviabiliza métodos manuais ou semiautomáticos em escala: cada peça tem forma, tamanho e contorno únicos. Os processos artesanais de mosaico preservam a estética, mas exigem mão de obra especializada cara e tempos longos de produção. Não existem, no Brasil, soluções automatizadas específicas para o reaproveitamento estético desses fragmentos.

## Objetivo (conforme contratado)

Desenvolver um sistema automatizado baseado em visão computacional e algoritmos de otimização para maximizar o aproveitamento de fragmentos irregulares de rochas ornamentais na produção de revestimentos tipo mosaico ou paginação — minimizando espaçamento entre fragmentos, otimizando a disposição espacial e reduzindo o desperdício de material.

## A solução em uma frase

Fotografa-se o retalho numa bancada calibrada; o software extrai o contorno real da peça em centímetros e calcula automaticamente o arranjo ótimo dos fragmentos num painel, gerando um mosaico executável que transforma resíduo em revestimento de alto valor.

## Pipeline (resumo — detalhes em [../docs/01-arquitetura.md](../docs/01-arquitetura.md))

1. **Captura controlada** — bancada com câmera perpendicular e iluminação LED controlada
2. **Calibração** — método de Zhang (tabuleiro de xadrez, OpenCV): correção de distorções ópticas
3. **Segmentação** — Segment Anything Model (SAM/ViT): máscaras binárias dos fragmentos
4. **Simplificação** — Douglas-Peucker: de ~11.000 pontos por contorno para centenas
5. **Dimensionamento** — correspondência pixel→cm via objeto de referência (+ LiDAR)
6. **Empacotamento** — Bottom-Left Fill + No-Fit Polygon + Simulated Annealing (problema NP-difícil)
7. **Saída** — arranjo otimizado do mosaico, pronto para execução física

## Produtos e serviços ofertados

1. Software de segmentação automática de fragmentos
2. Algoritmos de empacotamento inteligente para criação de revestimentos
3. Consultoria técnica para integração em processos industriais

## Status atual (19/07/2026 — M4)

| Frente | Situação |
|---|---|
| Equipamentos (M1) | Aquisições previstas: notebook, iPhone LiDAR, iluminação, bancada — conferir NFs arquivadas |
| Bancada de captura (M2–M3) | Entrega contratada até 14/07/2026 |
| Calibração Zhang (M2–M4) | **Em curso — entrega até 14/08/2026** |
| SAM + Douglas-Peucker (M4–M6) | **Em curso — otimização até 14/10/2026** |
| Algoritmos de empacotamento | Protótipos validados em escala reduzida (pré-projeto); integração contratada para M6–M7 |
| Execução financeira | Meta: ≥ 60% da 1ª parcela (R$ 10.800) comprometidos até o M5 |
