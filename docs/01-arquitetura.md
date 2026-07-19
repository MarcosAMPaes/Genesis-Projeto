# 01 — Arquitetura do Sistema

Pipeline completo, do fragmento físico ao mosaico executável. Cada estágio tem doc próprio com profundidade; aqui está o mapa.

## Visão do pipeline

```
Fragmento físico
   │  (1) Bancada: câmera perpendicular fixa + LED difuso controlado
   ▼
Foto bruta ──(2) Calibração + retificação──► Imagem fronto-paralela métrica   [doc 02]
   │  (3) SAM → máscara binária                                               [doc 03]
   ▼
Máscara ──(4) findContours ──► contorno denso (~10⁴ pontos)
   │  (5) Douglas-Peucker (ε calibrado)                                       [doc 03]
   ▼
Polígono simplificado ──(6) px→mm + validação Shapely──► Polígono métrico
   │  (7) Catálogo (SQLite → S3)                                              [doc 05]
   ▼
Conjunto de peças ──(8) Nesting: NFP + BLF + Simulated Annealing──►           [doc 04]
   ▼
Arranjo do painel ──(9) Export: render (SVG/PNG) + gabarito (PDF 1:1 / DXF)
```

## Contratos entre estágios (interfaces)

| Fronteira | Contrato |
|---|---|
| (2)→(3) | Imagem retificada fronto-paralela com fator de escala mm/px conhecido e registrado no metadado da sessão |
| (5)→(6) | Polígono simples (sem autointerseção), orientação CCW, fechado |
| (6)→(7) | Geometria em **mm**, `Polygon` Shapely válido, com `calib_profile_id` e `session_id` |
| (7)→(8) | Peças com buffer de junta/2 aplicável como transformação declarada (não destrutiva no catálogo) |
| (8)→(9) | Lista de `(piece_id, x_mm, y_mm, rotation_deg)` + métricas da solução + seed/params |

Regra de ouro: **pixel morre no estágio 6**. Empacotamento e exportação nunca veem pixels.

## Componentes de software (módulos do repositório)

```
petra/
├── calibration/   # Módulo A — Zhang, retificação, escala, validação dimensional
├── segmentation/  # Módulo B — SAM, máscara→polígono, simplificação
├── catalog/       # Módulo C — SQLite, ingest batch, sync S3
├── packing/       # Módulo D — NFP, BLF, SA, validador de colisão
├── export/        # Módulo F — SVG/PNG de aprovação + gabarito PDF 1:1/DXF
└── cli.py         # pipeline end-to-end: pasta de fotos → painel otimizado
ui/                # Módulo E — interface de operação (integra via CLI/API interna)
tests/             # unitários + suíte de benchmark versionada
```

Critérios de aceite por módulo: [07-roadmap.md](07-roadmap.md). Metas numéricas: [06-metricas-tecnicas.md](06-metricas-tecnicas.md).

## Infraestrutura e ambientes

| Recurso | Uso | Observação |
|---|---|---|
| Notebook Apple Silicon (16 GB) | Dev, calibração, segmentação unitária, SA | PyTorch via **MPS** — sem CUDA local; SAM ViT-H é pesado aqui, preferir variante leve (doc 03) |
| iPhone 15 Pro | Captura 48 MP + LiDAR (checagem de distância) | Foco/exposição travados por sessão (doc 02) |
| AWS EC2 g5.xlarge (A10G 24 GB) | Lotes de segmentação; corridas longas de SA multi-seed | ~120 h orçadas no ano — usar em lote, nunca ocioso |
| AWS S3 | Backup do catálogo (imagens, máscaras, polígonos) | 500 GB orçados; lifecycle para originais antigos |
| SQLite | Catálogo local (fonte de verdade) | S3 é espelho, não banco |

O mesmo código de segmentação roda local (MPS) e na nuvem (CUDA) — seleção de device por configuração, nunca dois códigos.

## Modos de operação

1. **Sessão de captura** — operador fotografa N fragmentos na bancada; ingest processa em lote até o catálogo (alvo: ≥ 60 fragmentos/h).
2. **Montagem de painel** — usuário define painel (dimensões, junta, subconjunto de peças por tipo/cor); sistema resolve o nesting e devolve arranjo + métricas.
3. **Exportação** — render para aprovação estética; gabarito 1:1 para execução física na marmoraria.

## Decisões em aberto (fechar e registrar aqui)

| # | Decisão | Critério | Status |
|---|---|---|---|
| D1 | Variante do SAM em produção (ViT-H × SAM 2 × MobileSAM × FastSAM) | IoU ≥ meta com menor custo (doc 03) | Aberta |
| D2 | Representação de colisão no nesting: NFP puro × raster híbrido | Robustez em concavidades + tempo (doc 04) | Aberta — NFP é a abordagem principal; raster é fallback de validação |
| D3 | Conjunto de rotações discretas por peça (0/90/180/270 × passos de 15–30°) | Qualidade × explosão do cache de NFP (doc 04) | Aberta |
| D4 | Formato de gabarito aceito pelas marmorarias (PDF 1:1 em tiles × DXF × projeção) | Validar com pilotos | Aberta |
| D5 | Junta-alvo padrão (mm) e tolerância de corte | Definir com marmoristas no 1º painel físico | Aberta |
| D6 | Stack da UI (desktop local × web local) | Operação offline na marmoraria pesa a favor de local | Aberta |
