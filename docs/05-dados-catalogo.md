# 05 — Modelo de Dados e Catálogo

Fonte de verdade: SQLite local. S3 é espelho/backup e insumo de processamento em nuvem. Regras: geometria em **mm float64 fim a fim** (arquitetura v2 — o engine `jagua-rs` opera em float robusto; conversão a inteiros só se o validador raster precisar), tudo rastreável até a foto e o perfil de calibração que o gerou.

> **Atualizado para a arquitetura v2 (doc 10, C2):** tabela `nfp_cache` removida (a detecção de colisão é responsabilidade do engine) e coluna `engine` adicionada a `packing_run`.

## 5.1 Esquema (SQLite)

```sql
CREATE TABLE calib_profile (
  id TEXT PRIMARY KEY,            -- hash do conteúdo
  created_at TEXT NOT NULL,
  device TEXT NOT NULL,           -- "iPhone17Pro/main-1x"
  K_json TEXT NOT NULL, dist_json TEXT NOT NULL,
  rms_px REAL NOT NULL,
  z_mm REAL,                      -- distância LiDAR câmera→plano
  scale_mm_px REAL NOT NULL,
  bench_config TEXT NOT NULL      -- id/hash da configuração física da bancada
);

CREATE TABLE session (
  id TEXT PRIMARY KEY,
  started_at TEXT NOT NULL,
  calib_profile_id TEXT NOT NULL REFERENCES calib_profile(id),
  background TEXT,                -- "verde-fosco" | "azul-fosco"
  thickness_mm REAL,              -- espessura da chapa da sessão (paralaxe, doc 02)
  operator TEXT, notes TEXT
);

CREATE TABLE fragment (
  id TEXT PRIMARY KEY,            -- ULID
  session_id TEXT NOT NULL REFERENCES session(id),
  rock_type TEXT,                 -- granito | mármore | quartzito | ...
  color_tag TEXT,                 -- catálogo estético (futuro objetivo do nesting)
  polygon_wkt TEXT NOT NULL,      -- POLYGON em mm, CCW, validado (gate doc 03 §3.5)
  area_mm2 REAL NOT NULL,
  bbox_w_mm REAL NOT NULL, bbox_h_mm REAL NOT NULL,
  n_points INTEGER NOT NULL,
  photo_path TEXT NOT NULL, mask_path TEXT NOT NULL,
  seg_model TEXT NOT NULL,        -- "sam2-large@<versão>"
  dp_epsilon_mm REAL NOT NULL,
  status TEXT NOT NULL DEFAULT 'available',  -- available|reserved|used|rejected
  created_at TEXT NOT NULL
);

CREATE TABLE packing_run (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  panel_w_mm REAL NOT NULL, panel_h_mm REAL NOT NULL,
  joint_mm REAL NOT NULL,
  objective TEXT NOT NULL,        -- panel|strip
  engine TEXT NOT NULL,           -- "sparrow@<versão>" | "petra-panel@<versão>"
  params_json TEXT NOT NULL,      -- config do engine + camada painel (subconjunto/SA)
  seed INTEGER NOT NULL,
  code_version TEXT NOT NULL,     -- git hash
  env_fingerprint TEXT NOT NULL,  -- arch+threads+versões pinadas (determinismo escopado, doc 10)
  utilization REAL, height_mm REAL, n_placed INTEGER,
  runtime_s REAL, validated INTEGER NOT NULL  -- validador independente passou
);

CREATE TABLE placement (
  run_id TEXT NOT NULL REFERENCES packing_run(id),
  fragment_id TEXT NOT NULL REFERENCES fragment(id),
  x_mm REAL NOT NULL, y_mm REAL NOT NULL, rotation_deg REAL NOT NULL,
  PRIMARY KEY (run_id, fragment_id)
);
```

## 5.2 Formatos e convenções

- **WKT** para geometria no banco (legível, Shapely nativo: `shapely.from_wkt`); GeoJSON/JSON do jagua-rs como formatos de exportação/intercâmbio;
- Unidade canônica **mm float64** fim a fim (v2); rasterização/inteiros apenas dentro do validador independente, se necessário;
- IDs: ULID (ordenáveis por tempo);
- `status` do fragmento controla o ciclo de vida físico: peça usada num mosaico real sai do estoque (`used`) — o catálogo é também **inventário do estoque de retalhos**;
- Mídia (foto/máscara) no filesystem organizado por sessão: `data/sessions/<session_id>/{raw,rect,mask}/<fragment_id>.png` — path no banco, binário fora do banco.

## 5.3 Sincronização S3

- Layout espelho: `s3://<bucket>/petra/{db-backups,sessions}/...`;
- `db-backups/`: snapshot diário do SQLite (o banco é pequeno; o volume está nas imagens);
- Upload de sessão ao fim do ingest (não por arquivo); lifecycle: originais RAW → classe fria após 90 dias (500 GB orçados — monitorar);
- Nuvem processa, local decide: jobs na EC2 leem de S3 e devolvem máscaras/polígonos; o merge no catálogo acontece localmente com o mesmo gate de validação.

## 5.4 Reprodutibilidade e proveniência

Cadeia completa reconstruível para qualquer painel produzido:

```
painel físico → packing_run (seed, params, git hash) → placements → fragments
             → sessions → calib_profile → fotos de calibração
```

Isso é simultaneamente disciplina de engenharia e a evidência técnica de execução do projeto (relatórios em `/gestao`).

## 5.5 Conjunto de validação (asset versionado)

`data/validation/`: ≥ 30 fragmentos com máscara anotada manualmente + 10 fragmentos com medidas de paquímetro. É o gabarito das métricas do doc 06 — versionado, cresce com casos difíceis reais (doc 03 §3.6), nunca é usado para ajustar parâmetros e avaliar ao mesmo tempo (separar desenvolvimento/validação quando começar a otimizar ε e modelo). É a única exceção à regra de binários fora do Git: pode usar Git LFS dentro do orçamento documentado no manifesto; PR não baixa blobs, e workflows de modelo/benchmark/físico fazem fetch filtrado. Dados operacionais, sessões reais completas e pesos continuam exclusivamente fora do Git/S3. Se a quota LFS não comportar o corpus, o manifesto aponta os objetos no S3 por origem e SHA-256.
