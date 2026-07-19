# 06 — Métricas Técnicas e Protocolos de Medição

Metas numéricas por estágio + como medir. Benchmarks rodam a cada release; resultados versionados em `tests/benchmarks/results/`. Métricas de projeto/negócio: [`/gestao/11-metricas-projeto.md`](../gestao/11-metricas-projeto.md).

## Calibração e metrologia (doc 02)

| Métrica | Meta | Protocolo |
|---|---|---|
| Erro de reprojeção (RMS) | < 0,5 px | Saída do `calibrateCamera`, ≥ 20 poses; inspecionar resíduo por pose e descartar outliers |
| Erro dimensional absoluto | ≤ 2 mm (alvo 1 mm) | 10 peças de referência, 2+ eixos por peça, paquímetro × sistema; reportar médio/máximo/por eixo |
| Estabilidade entre sessões | variação < 0,5% na escala | Padrão ArUco re-medido em 5 sessões distintas |
| Consistência LiDAR × calibração | divergência < 2% em Z | Comparação automática por sessão; divergência = alerta de recalibração |
| Erro de paralaxe residual | coberto pela meta dimensional | Validar com peças de espessuras distintas (10/20/30 mm) |

## Segmentação (doc 03)

| Métrica | Meta | Protocolo |
|---|---|---|
| IoU máscara × anotação manual | > 0,95 (global e por classe de rocha) | Conjunto de validação (≥ 30 anotados); reportar pior classe, não só média |
| Fidelidade de borda (Hausdorff) | < 2 mm | Contorno predito × anotado, em mm |
| Taxa sem intervenção | > 90% | % de capturas de sessão real sem prompt manual |
| Tempo por fragmento | < 30 s local (MPS) · < 5 s nuvem (A10G) | Benchmark com 20 imagens padrão, mediana |
| Rejeições corretas | 100% dos casos de gate (§3.5) logados | Auditoria da fila de rejeição |

## Simplificação (doc 03 §3.4)

| Métrica | Meta | Protocolo |
|---|---|---|
| Redução de pontos | > 95% (~10⁴ → 100–400) | Contagem antes/depois |
| Desvio de área | < 1% | área(simplificado) ÷ área(máscara) |
| Desvio máximo local | ≤ ε = 0,5 mm (por construção DP) | Verificação amostral + curva ε publicada no benchmark |

## Empacotamento (doc 04)

| Métrica | Meta | Protocolo |
|---|---|---|
| Sobreposições | **0 — invariante** | Validador independente (Shapely/raster) em toda solução aceita; falha = bug bloqueante |
| Contenção no painel + junta ≥ g | 100% das peças | Mesmo validador |
| Aproveitamento (painel) | ≥ 70% (*fixar após 1º benchmark real*) | Σ área peças (sem buffer) ÷ área do painel; benchmark interno congelado |
| Ganho sobre baseline | SA > BLF-área-decrescente, estatisticamente (10 seeds) | Mesmo conjunto, mediana e melhor-de-10 |
| Tempo de convergência | < 10 min (painel 1 m², ~50 peças, notebook) | Benchmark padrão, orçamento de tempo fixo |
| Reprodutibilidade | bit a bit por seed | Teste automatizado no CI |
| Sanidade externa | Aproveitamento na faixa publicada p/ SA em instâncias ESICUP | shirts/swim/trousers, comparação qualitativa |

## Sistema integrado

| Métrica | Meta | Protocolo |
|---|---|---|
| Throughput da bancada | ≥ 60 fragmentos/h (foto → catálogo) | Cronometragem de sessão real de ≥ 1 h |
| Pipeline end-to-end | verde semanalmente | Teste de fumaça: pasta de fotos padrão → painel validado, no CI local |
| Fechamento físico do gabarito | desvio de montagem ≤ junta/2 | Painel-teste interno: montar e medir vs. layout digital (antes do 1º piloto) |
| Tamanho do catálogo | ≥ 500 fragmentos reais antes dos pilotos | Contagem no banco |

## Regras

1. Meta sem protocolo não é meta — toda linha acima tem os dois; novas métricas seguem o padrão.
2. *Baseline a fixar* (aproveitamento): primeira medição real define a meta contratual interna; não chutar.
3. Resultado de benchmark acompanha: git hash, seed(s), hardware, data — sem isso não entra no histórico.
4. Piora vs. release anterior = regressão: investigar antes de seguir (qualidade ou tempo).
