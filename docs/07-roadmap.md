# 07 — Roadmap de Desenvolvimento

Ordem de construção dos módulos, entregas e critérios de aceite. Datas-alvo alinhadas ao calendário do projeto (referência externa: `/gestao/03-cronograma.md`), mas este doc só trata do software.

## Sequência e dependências

```
A Calibração ──► B Segmentação ──► C Catálogo ──► D Empacotamento ──► F Export
                                        │                               │
                                        └────────── E Interface ────────┘
```

D pode começar antes de C estar completo (usar polígonos sintéticos + ESICUP), mas só "conta" rodando sobre peças reais do catálogo.

## Módulo A — Calibração e metrologia (agora; alvo: meados de ago/2026)

1. `calibrate.py`: sessão assistida de ≥ 20 poses → perfil versionado (`calib_profile`)
2. Retificação fronto-paralela por ArUco/ChArUco + escala mm/px por sessão (doc 02 §2.3)
3. Correção de paralaxe por espessura (t/Z com Z do LiDAR)
4. `validate_dims.py`: protocolo paquímetro × sistema com relatório

**Aceite:** RMS < 0,5 px; erro dimensional ≤ 2 mm nas 10 peças de referência; escala estável < 0,5% entre 5 sessões.

## Módulo B — Segmentação (agora → meados de out/2026)

1. Pipeline retificada → SAM → máscara → gate de validade → polígono em mm
2. Comparativo de modelos (D1): SAM × SAM 2 × MobileSAM × FastSAM no conjunto de validação
3. Estudo do ε (curva ε × pontos × desvio de área × Hausdorff); padrão ε = 0,5 mm confirmado ou revisado
4. Execução dual local/nuvem (mesmo código, device por config)
5. Conjunto de validação anotado (≥ 30) montado e versionado

**Aceite:** IoU > 0,95 (pior classe); > 90% sem intervenção; tempos dentro da meta; decisão D1 registrada.

## Módulo C — Catálogo (paralelo a B; alvo: meados de out/2026)

1. Esquema SQLite do doc 05 + migrações
2. Ingest batch (pasta → catálogo) com gate de validade e fila de rejeição
3. Sync S3 por sessão + snapshot diário do banco + monitor de custo

**Aceite:** ingest ≥ 60 fragmentos/h; ≥ 500 fragmentos reais catalogados antes dos pilotos; restauração de backup testada.

## Módulo D — Empacotamento (set → meados de nov/2026) — módulo crítico

1. Geometria base: µm inteiros, buffer de junta (Clipper), decomposição convexa
2. NFP por Minkowski + decomposição (doc 04 §4.3) com cache persistente; suíte de degenerados
3. IFP + BLF com critérios de posição configuráveis; baseline área-decrescente
4. Validador independente de colisão/contenção (invariante)
5. SA: sequência+rotação, vizinhanças, T₀ auto-calibrado, α=0,995, melhor-global, multi-seed
6. Benchmark interno congelado + sanidade ESICUP
7. Fallback raster (§4.6) como validador barato

**Aceite:** 0 sobreposições em toda a suíte; SA > baseline em 10 seeds; < 10 min no benchmark padrão; reprodutível por seed.

## Módulo E — Interface e controle (especificar já; construir out/2026 → jan/2027)

1. Especificação funcional **antes** de contratar o desenvolvimento externo: telas de sessão de captura (com correção por clique = point prompt), revisão de catálogo, montagem de painel (dimensões, junta, filtros de peça), acompanhamento da otimização, exportação
2. Integração UI ↔ módulos via CLI/API interna (a UI nunca implementa lógica de pipeline)
3. Fallback interativo de segmentação (§3.2) exposto ao operador

**Aceite:** operador não programador executa capturar → revisar → empacotar → exportar sem tocar em código.

## Módulo F — Exportação (nov/2026 → fev/2027)

1. Render SVG/PNG do painel (aprovação estética; cores/texturas reais das fotos das peças)
2. Gabarito de montagem **1:1 em tiles A3/A0** (PDF) com marcas de registro + numeração de peças; DXF opcional (D4)
3. Relatório por painel: peças, aproveitamento, área, junta, proveniência (run/seed)
4. Painel-teste interno montado fisicamente antes do 1º piloto (métrica de fechamento físico, doc 06)

**Aceite:** mosaico físico montado a partir do gabarito com desvio ≤ junta/2.

## Regras transversais

1. `main` sempre com pipeline end-to-end verde (teste de fumaça semanal);
2. Benchmark + changelog por release; regressão bloqueia;
3. Decisões D1–D6 fechadas viram registro no doc 01 (data + critério usado);
4. Dados fora do git (S3); código, esquemas, curvas e resultados de benchmark dentro;
5. Todo resultado relevante (curva do ε, comparativo SAM, ganho do SA) sai em formato exportável — vira evidência nos relatórios de `/gestao` sem retrabalho.
