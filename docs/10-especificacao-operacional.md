# 10 — Especificação Operacional Consolidada (Arquitetura v2)

**Status: fonte de verdade da implementação.** Consolida docs 01–09 e a pesquisa de estado da arte. Onde houver conflito com docs anteriores, **este documento prevalece** (conflitos conhecidos listados na §4). Decisão estrutural incorporada: **`jagua-rs`/`sparrow` no fluxo principal do empacotamento** (doc 09); NFP própria deixa de ser plano e vira apenas fundamento teórico (doc 04).

---

## 1. Especificação operacional

### 1.1 Atores

| Ator | Descrição | Interage via |
|---|---|---|
| Operador de captura | Funcionário da marmoraria/equipe; fotografa peças na bancada | UI (Módulo E) |
| Projetista de painel | Define painel, junta, filtros de peças; aprova arranjo | UI |
| Marmorista | Executa o mosaico físico a partir do gabarito | Gabarito 1:1 (papel) |
| Desenvolvedor | Mantém pipeline; roda lotes e benchmarks | CLI |
| Sistema (batch) | Ingest, sync S3, jobs de nuvem | CLI/cron |

### 1.2 Modos de operação

1. **Sessão de captura** — operador calibra (se necessário), fotografa N peças, revisa segmentações, confirma ingest no catálogo. Meta: ≥ 60 peças/h.
2. **Montagem de painel** — projetista escolhe painel (L×A), junta g, filtros (tipo/cor/espessura); sistema resolve o nesting; projetista aprova ou re-roda com outros parâmetros.
3. **Exportação e execução** — render aprovado → gabarito 1:1 + relatório; peças usadas mudam para `used` no estoque.
4. **Lote/nuvem** — reprocessamento de sessões, corridas longas multi-seed, backup.

### 1.3 Ambiente e restrições operacionais

- Operação **100% funcional offline** no notebook (marmoraria sem internet confiável); nuvem é aceleração, nunca dependência (RNF-06);
- Bancada: câmera fixa perpendicular, LED difuso, fundo fosco trocável (verde/azul), placa ArUco no plano da face superior das peças;
- Peças: fragmentos de rocha 5–80 cm, espessura 5–40 mm, potencialmente côncavos; nunca empilhadas nem se tocando na mesa;
- Orçamento de nuvem: ~120 h GPU + 500 GB S3 no ano — jobs em lote, monitor de custo obrigatório (RNF-07).

### 1.4 Fluxo principal (arquitetura v2)

```
Foto (iPhone, sessão calibrada)
  → [A] undistort + retificação ArUco + escala mm/px (+ correção espessura t/Z)
  → [B] segmentação (bake-off: BiRefNet × SAM 3 × SAM 2 leve) → máscara → contorno
        → Douglas-Peucker (ε=0,5 mm) → gate de validade → POLÍGONO MÉTRICO (mm)
  → [C] catálogo SQLite (estoque + proveniência) ⇄ espelho S3
  → [D] EMPACOTAMENTO:
        strip  → sparrow (via spyrrow)                      [SOTA pronto]
        painel → camada própria (seleção de subconjunto + SA) sobre sparrow/jagua-rs
        junta  → min_item_separation nativo do engine
        sempre → validador independente (Shapely/raster) = gate de saída
  → [F] render SVG/PNG → gabarito 1:1 em tiles + relatório do painel
  → [E] UI cobre os modos 1–3 chamando os módulos via CLI/API interna
```

---

## 2. Contratos de entrada e saída — Módulos A–F

Convenções globais: unidade canônica **mm** (float64) a partir da saída de A; polígonos **simples, CCW, fechados, válidos (Shapely)**; todo artefato carrega `schema_version`; IDs ULID; timestamps ISO-8601 UTC.

### Módulo A — Calibração e metrologia

| | Contrato |
|---|---|
| **Entrada (calibração)** | ≥ 20 fotos do tabuleiro (mesma lente/foco); dimensão certificada do quadrado (mm) |
| **Saída (calibração)** | `calib_profile.json`: `{id, device, lens, K[3][3], dist[5], rms_px, img_size, z_mm_lidar, created_at, bench_config_hash}` — **rejeitar se** `rms_px ≥ 0.5` |
| **Entrada (sessão)** | Fotos brutas + `calib_profile_id` + espessura(s) declarada(s) da sessão + placa ArUco visível |
| **Saída (sessão)** | Por foto: imagem retificada fronto-paralela + `session_meta.json`: `{session_id, calib_profile_id, scale_mm_px, homography[3][3], aruco_ids[], thickness_mm, background, residual_check_mm}` |
| **Erros** | ArUco < 4 marcadores → rejeita foto; `residual_check_mm > 1.0` (placa re-medida) → bloqueia sessão; divergência LiDAR×calibração > 2% → alerta de recalibração |
| **Invariante** | Nenhuma imagem segue para B sem `scale_mm_px` validado na própria cena |

### Módulo B — Segmentação e extração de polígono

| | Contrato |
|---|---|
| **Entrada** | Imagem retificada + `session_meta` + modo (`auto` \| `prompt{points/box/concept}`) |
| **Saída** | Por peça: `fragment_geom.json`: `{fragment_id, polygon_mm: [[x,y],...], area_mm2, bbox_mm, n_points, quality_warnings, seg_model, seg_score, dp_epsilon_mm, mask_path}` |
| **Gate (rejeição com log)** | Máscara toca borda da imagem; área fora de [25 cm², 1 m²]; >1 componente grande; polígono inválido após `buffer(0)`; `area_poly/area_mask ∉ [0.99, 1.01]` |
| **Pós-processamento fixo** | Maior componente → fill holes → morfologia leve (≤5 px) → `findContours(RETR_EXTERNAL, CHAIN_APPROX_NONE)` → `approxPolyDP(epsilon_px)` → mm. **Conversão do ε:** `epsilon_px = ε_mm / scale_mm_px` (com ε_mm = 0,5 e `scale_mm_px` em mm/px; o `epsilon` do OpenCV está na unidade das coordenadas do contorno, i.e. pixels) |
| **Invariante** | 3–5000 vértices distintos por peça (4–5001 coordenadas no anel fechado); 100–1000 é faixa esperada informativa. Contagens 3–99 e 1001–5000 são aceitas com `VERTEX_COUNT_BELOW_EXPECTED` e `VERTEX_COUNT_ABOVE_EXPECTED`, respectivamente; Hausdorff(contorno, polígono) ≤ ε por construção |

### Módulo C — Catálogo

| | Contrato |
|---|---|
| **Entrada** | `fragment_geom.json` + mídia + metadados de sessão; comandos de ciclo de vida (`reserve`, `use`, `reject`, `release`) |
| **Saída** | SQLite (esquema doc 05 **atualizado** — ver C2 e L4): consultas por filtro `{rock_type, color_tag, thickness_mm, area range, status='available'}` → lista de peças com geometria |
| **Garantias** | Transacional; peça `used` nunca reaparece em consulta de montagem; toda peça referencia `session_id` e `calib_profile_id` (proveniência completa) |
| **Sync** | Espelho S3 por sessão + snapshot diário do banco; restauração testada (TC-3) |

### Módulo D — Empacotamento (v2: sparrow/jagua-rs no fluxo principal)

| | Contrato |
|---|---|
| **Entrada** | `panel_spec.json`: `{mode: 'strip'\|'panel', width_mm, height_mm?, joint_mm, pieces: [fragment_id...], rotation: 'free'\|'discrete[...]'\|'per-piece', time_budget_s, seed}` |
| **Saída** | `solution.json`: `{run_id, placements: [{fragment_id, x_mm, y_mm, rotation_deg}], metrics: {utilization, height_mm, n_placed, runtime_s}, engine: {name, version, params}, seed, code_version, validated: true}` |
| **Caminho strip** | `spyrrow`/sparrow direto (fase exploração+compressão; warm-start suportado); junta via `min_item_separation = joint_mm` |
| **Caminho painel** | Camada própria: seleção de subconjunto (ordenação por valor/densidade + SA sobre a composição) orquestrando o solver; rota padrão **P2** (iterar strip com largura do painel; aceitar subconjunto cuja altura ≤ altura do painel), upgrade **P1** (modo contêiner-fixo nativo no fork do sparrow) — decisão D7 |
| **Gate de saída (inegociável)** | Validador independente (Shapely; tolerância 0,01 mm²): zero interseção par a par, contenção total, distância mínima ≥ `joint_mm`. `validated=false` → solução não sai do módulo (RF-15) |
| **Invariantes** | Determinismo **escopado**: mesmo `(input, seed)` → mesma solução **no ambiente de referência pinado** (versões de engine/deps, arquitetura de CPU, nº de threads fixado); entre ambientes distintos exige-se **equivalência geométrica** (mesmas métricas ±0,1% e solução válida), não bits idênticos. Peças de entrada = `available`; solução aceita reserva as peças |

### Módulo E — Interface de operação

| | Contrato |
|---|---|
| **Entrada** | Ações do operador/projetista sobre os modos 1–3 (§1.2) |
| **Saída** | Chamadas aos módulos **exclusivamente via CLI/API interna** (a UI não contém lógica de pipeline); correção interativa de segmentação = re-chamada de B com `prompt` |
| **Aceite** | Operador não programador completa capturar → revisar → montar → exportar sem tocar em código (RF-20) |
| **Pré-condição de contratação** | Especificação funcional escrita (telas, estados, erros) **antes** do desenvolvimento terceirizado — lacuna L6 |

### Módulo F — Exportação

| | Contrato |
|---|---|
| **Entrada** | `solution.json` validado + mídia das peças |
| **Saída** | (i) `render.svg/png` com textura real das peças; (ii) `gabarito.pdf` **1:1** em tiles A3/A0 com marcas de registro, numeração de peça e orientação; (iii) `relatorio.json/pdf`: aproveitamento, área, junta, lista de peças, proveniência `(run_id, seed, code_version)` |
| **Invariante** | Dimensões no PDF conferem 1:1 (teste de impressão com régua — TA-F2); IDs no gabarito = IDs físicos etiquetados nas peças |

---

## 3. Matriz de requisitos e rastreabilidade

Origens: `AII` = Anexo II contratado (seção), `CRON-n` = entrega n do cronograma físico, `E-x.y` = item do edital, `D0x` = doc técnico, `SOTA` = doc 09. Cada requisito tem teste nomeado (registro completo em §7.1).

> **Auditabilidade:** os documentos primários (`AII`, `CRON`, `E-*`) estão versionados em [`/gestao/fontes/`](../gestao/fontes/) — Edital 09/2025, Anexo II contratado e Resultado Preliminar — com integridade garantida por `SHA256SUMS.txt`. A rastreabilidade é verificável dentro do próprio repositório.

### Requisitos funcionais

| ID | Requisito | Origem | Módulo | Aceite | Teste |
|---|---|---|---|---|---|
| RF-01 | Calibrar câmera (Zhang) e persistir perfil versionado | AII-Descrição; CRON-3 | A | RMS < 0,5 px | TA-1 |
| RF-02 | Corrigir distorção em toda captura automaticamente | AII-Descrição | A | 100% das fotos | TA-2 |
| RF-03 | Retificação fronto-paralela + escala mm/px por sessão (ArUco) | D02 §2.3 | A | `residual_check_mm ≤ 1.0` | TA-3 |
| RF-04 | Corrigir paralaxe por espessura (t/Z, LiDAR) | D02 §2.3 | A | Erro ≤ 2 mm c/ espessuras 10–40 mm | TA-4 |
| RF-05 | Segmentar múltiplas peças por foto sem intervenção | AII-Descrição (SAM); SOTA | B | > 90% sem prompt | TB-1 |
| RF-06 | Correção interativa por prompt (ponto/caixa) | D03 §3.2 | B/E | Caso difícil resolvido em ≤ 3 cliques | TB-2 |
| RF-07 | Simplificar contorno (Douglas-Peucker, ε físico) | AII-Descrição; CRON-4 | B | ~10⁴→10²–10³ pts; desvio área < 1% | TB-3 |
| RF-08 | Gate de validade geométrica na entrada do catálogo | D03 §3.5 | B/C | 0 polígonos inválidos no banco | TB-4 |
| RF-09 | Catálogo persistente com proveniência e estoque | AII-Descrição (banco S3) | C | Cadeia painel→foto reconstruível | TC-1 |
| RF-10 | Espelho S3 + snapshot + restauração | AII-Orçamento (S3) | C | Restauração íntegra em teste | TC-3 |
| RF-11 | Nesting strip (largura fixa, min altura) | AII-Objetivo ("altura total") | D | sparrow; ≥ resultados publicados em ESICUP | TD-1 |
| RF-12 | Nesting painel: subconjunto de peças únicas em painel L×A fixo | AII-Objetivo (mosaico/paginação) | D | Utilization ≥ baseline BLF + margem | TD-2 |
| RF-13 | Junta paramétrica g uniforme | D04 §4.2; pilotos | D | Distância mínima ≥ g em 100% dos pares | TD-3 |
| RF-14 | Rotação contínua; restrição por peça opcional (veio direcional) | SOTA (jagua-rs); D8 | D | Config por peça respeitada | TD-4 |
| RF-15 | Validador independente de colisão/contenção | D01 princípio 2 | D | 0 falsos aceites na suíte adversarial | TD-5 |
| RF-16 | Determinismo por seed no ambiente de referência; equivalência geométrica entre ambientes | D01 princípio 3 | D | Bit a bit no CI (ambiente pinado); métricas ±0,1% fora dele | TD-6 |
| RF-17 | Render estético com textura real | AII-Produtos | F | Aprovação em piloto | TF-1 |
| RF-18 | Gabarito 1:1 em tiles com registro e numeração | D07 Módulo F | F | Impressão confere com régua (±1 mm/m) | TF-2 |
| RF-19 | Relatório por painel (aproveitamento, proveniência) | E-13 (evidências) | F | Gerado para 100% dos painéis | TF-3 |
| RF-20 | UI: fluxo completo por operador leigo | AII-Orçamento (interface); CRON-6 | E | Teste de usabilidade com operador real | TE-1 |
| RF-21 | CLI batch end-to-end (pasta → painel validado) | D01 | A–F | Teste de fumaça semanal verde | TS-1 |

### Requisitos não funcionais

| ID | Requisito | Origem | Aceite | Teste |
|---|---|---|---|---|
| RNF-01 | Erro dimensional ≤ 2 mm (alvo 1) | D06 | 10 peças × paquímetro | TA-4 |
| RNF-02 | IoU > 0,95 na pior classe de rocha | D06 | Conjunto validação ≥ 30 | TB-1 |
| RNF-03 | Hausdorff contorno ≤ 2 mm | D06 | Idem | TB-1 |
| RNF-04 | Painel 1 m²/~50 peças < 10 min (notebook) | D06 | Benchmark congelado | TD-7 |
| RNF-05 | Throughput ≥ 60 peças/h | D06 | Sessão real cronometrada | TS-2 |
| RNF-06 | Operação offline completa | §1.3 | Teste com rede desligada | TS-3 |
| RNF-07 | Nuvem dentro de 120 h GPU + 500 GB/ano | Orçamento | Monitor mensal | TS-4 |
| RNF-08 | Licenças de dependências compatíveis com o modelo de negócio | D9/L8 | Auditoria de licenças no CI | TS-5 |
| RNF-09 | Resultados exportáveis como evidência (relatórios do projeto) | E-13 | Benchmark → PDF/JSON | TF-3 |

---

## 4. Contradições e lacunas documentais

### Contradições (C) — com resolução

| # | Contradição | Resolução |
|---|---|---|
| C1 | Doc 04 planeja **NFP própria**; doc 09 recomenda **sparrow/jagua-rs** | **Resolvida por esta spec:** engine externo no fluxo principal; doc 04 = teoria + fallback. Gate G1 (§7) confirma com dados |
| C2 | Docs 01/04/05 fixavam convenção **µm inteiros (Clipper)**; jagua-rs opera em float robusto | Nova convenção: **mm float64** fim a fim; inteiros só se o validador raster precisar. `nfp_cache` **eliminado** (o engine cuida de colisão). **Doc 05 já atualizado** nesta revisão — esquema sem `nfp_cache`, convenção mm float64, coluna `engine` em `packing_run` |
| C3 | D3 (doc 01) discretiza rotações p/ cache de NFP; jagua-rs suporta **rotação contínua** | D3 dissolvida → substituída por D8 (restrição de rotação por razão *estética*, não computacional) |
| C4 | Cronograma contratado (interface M8–M9, pilotos M9–M11) × calendário operacional (1 mês antes) | Operacional = plano interno; contratado = referência de cobrança. Relatórios sempre contra o contratado (gestao/03) |
| C5 | Anexo II cabeçalho "Edital 03/2025 / 2024-3BN5Z" × edital vigente 09/2025 / 2024-0RDMG | Usar número do processo do TO em toda correspondência; conferir no SIGFAPES (gestao/01) |
| C6 | Orçamento aprovado: **iPhone 15 Pro**; hardware real **confirmado: iPhone 17 Pro** (19/07/2026) | Docs técnicos atualizados para o 17 Pro; texto do item aprovado preservado em gestao/04 com nota de equivalência. Pendente: anexar justificativa de equivalência funcional (LiDAR + 48 MP) à NF na prestação de contas; se a NF divergir materialmente do aprovado, consultar FAPES antes (gestao/09) |
| C7 | Proposta promete "algoritmos proprietários" como barreira; v2 adota engine MIT/MPL | Valor proprietário reposicionado (doc 09 §3): metrologia, variante painel, estética, fechamento físico. Registrar assim no relatório técnico e no discurso comercial |
| C8 | Repo **GPL-3.0** × modelo de negócio de **licenciamento** da tecnologia | Precisão jurídica: GPL **permite uso comercial** e o copyleft só se ativa na **distribuição** do software — quem recebe binário/código tem direito ao fonte sob GPL; uso interno ou como serviço não obriga publicação (isso seria AGPL). O risco real é específico: distribuir o sistema a marmorarias sob GPL daria a elas direito de redistribuir o fonte, enfraquecendo licenciamento proprietário. Como a empresa é titular do copyright do código próprio, **dual licensing** é viável (GPL pública + licença comercial), desde que dependências (MIT/MPL-2.0 — compatíveis) e eventuais contribuições externas (exigir CLA) sejam geridas. → Decisão D9 |

### Lacunas (L) — com ação e prazo

| # | Lacuna | Ação | Quando |
|---|---|---|---|
| L1 | Junta padrão g e tolerância de corte indefinidas | Definir com marmorista no 1º painel-teste (D5) | S5 |
| L2 | Formato de gabarito aceito pelas marmorarias não validado | Levar PDF 1:1 e DXF ao 1º piloto (D4) | S5–S6 |
| L3 | Conjunto de validação anotado ainda não existe | Construir (≥ 30 máscaras + 10 peças com paquímetro). **Desvio deliberado da regra geral de dados fora do Git:** somente esse corpus pequeno, congelado e auditável pode usar Git LFS; dados operacionais, sessões completas e pesos permanecem no S3/fora do Git. O manifesto registra backend, SHA-256, tamanho, splits e atributos; PR usa `GIT_LFS_SKIP_SMUDGE=1`; apenas workflows `model`/`benchmark`/`physical` fazem fetch filtrado de `data/validation/**`. Antes do primeiro blob real, definir `lfs_budget_bytes = 80%` da quota disponível; acima do orçamento, usar manifesto+hash com origem S3. | S1–S2 |
| L4 | Espessura por peça: esquema atual só tem por sessão | Adicionar `thickness_mm` opcional em `fragment` (herda da sessão se nulo) | S2 |
| L5 | Licença do SAM 3 ("SAM License") não auditada para uso comercial; checkpoint SAM 3.1 sem caminho público reproduzível para inferência de imagem (facebookresearch/sam3#526) | Runtime falha fechado e permanece científico: `license_approved=false`, Linux/CUDA e concept prompt no contrato; não preencher revisão de código/PyTorch/imagem até um smoke oficial reproduzível. Auditar antes de produção; BiRefNet (MIT) e SAM 2.1 (Apache-2.0) seguem no bake-off | S2 (bake-off) |
| L6 | Especificação funcional da UI inexistente (pré-requisito do contrato de terceiros) | Escrever `docs/11-espec-ui.md` | S4 |
| L7 | Padrões de painel (dimensões comerciais) não definidos | Levantar com parceiros (30×30? 60×60? livre?) | S4–S5 |
| L8 | Auditoria de licenças de dependências (RNF-08) não automatizada | `pip-licenses`/`cargo-license` no CI | S0 |
| L9 | Protocolo de aceite físico com marmoraria (quem mede, como registra) | Anexar ao plano de piloto | S5 |
| L10 | CI não configurado (repo só tem LICENSE/README) | GitHub Actions: lint + testes + benchmark local | S0 |

---

## 5. Decisões pendentes (registro vivo)

| ID | Decisão | Opções | Critério | Prazo | Status |
|---|---|---|---|---|---|
| D1 | Modelo de segmentação padrão | BiRefNet × SAM 3 × SAM 2 leve × chroma baseline | IoU pior classe + Hausdorff mm + custo local | S2 | **Aberta** (bake-off TB-1) |
| D2 | Engine de empacotamento | ~~NFP própria~~ × sparrow/jagua-rs | Gate G1 (benchmark ESICUP + instâncias "tipo retalho") | Fim de S0 | **Default arquitetural: sparrow/jagua-rs — condicionado à aprovação de G1** (só reverte ao fallback do doc 04 se G1 reprovar; após G1 aprovado, status vira "decidida") |
| D3 | ~~Rotações discretas p/ cache NFP~~ | — | — | — | **Dissolvida** (rotação contínua nativa) |
| D4 | Formato do gabarito | PDF 1:1 tiles × DXF × ambos | Aceitação no piloto | S6 | Aberta |
| D5 | Junta padrão g (mm) | 2–10 mm típico | Estética + tolerância de corte | S5 | Aberta |
| D6 | Stack da UI | Desktop local (Tauri/PySide) × web local | Offline obrigatório (RNF-06) + custo do terceiro | S4 | Aberta |
| D7 | Variante painel: rota de implementação | **P2** (orquestrar strip por iteração de subconjunto, Python) × P1 (modo contêiner fixo em fork Rust do sparrow) | TD-2: qualidade e tempo de P2; P1 só se P2 insuficiente | S3–S4 | Aberta (default P2) |
| D8 | Restrição de rotação estética por peça (veio direcional) | Livre × faixas por peça | Feedback de projetista no piloto | S6 | Aberta |
| D9 | Licença/publicidade do repo × modelo de licenciamento comercial | GPL público × privado até INPI × dual licensing | Estratégia de PI (registro INPI ~S7) | S6 | Aberta |

---

## 6. Riscos técnicos (v2)

| # | Risco | P | I | Mitigação | Gatilho/teste |
|---|---|---|---|---|---|
| R1 | Metrologia em campo: erro dimensional > 2 mm (paralaxe, ArUco fora do plano, deriva) | Alta | Alto | Placa no plano da face superior; correção t/Z; `residual_check` por sessão; TA-4 semanal | TA-4 falha |
| R2 | Segmentação falha em classes específicas (rocha escura, reflexo, translúcida) | Média | Alto | Bake-off por pior classe; 2 fundos; prompt interativo; conjunto de validação vivo | TB-1 por classe < 0,95 |
| R3 | **Variante painel (camada própria) mais difícil que o previsto** — agora o principal risco de engenharia | Média | Alto | Rota P2 primeiro (sem Rust); P1 como upgrade; gate G3 com critério objetivo; fallback documentado (doc 04) | TD-2 reprova em S4 |
| R4 | Interop Rust↔Python (spyrrow imaturo, casos não cobertos) | Média | Médio | spyrrow para strip; subprocess+JSON como caminho universal; pin de versões; testes de contrato TD-8 | TD-8 falha em upgrade |
| R5 | sparrow subótimo no nosso perfil de instância (poucas peças grandes e côncavas vs. têxtil) | Baixa–Média | Médio | G1: benchmark com instâncias "tipo retalho" na semana 1; comparar com LBF baseline | G1 < meta |
| R6 | SAM/BiRefNet lentos no Apple Silicon (MPS) | Média | Médio | BiRefNet é leve; lotes na g5.xlarge; medir em TB-5 | > 30 s/peça local |
| R7 | Estouro do orçamento de nuvem | Média | Médio | Monitor TS-4 mensal; processamento em lote agendado | Gasto > pró-rata |
| R8 | Gabarito não fecha na montagem física (acúmulo junta+tolerância) | Média | Alto | Painel-teste interno antes do piloto (G4); junta como margem; protocolo L9 | G4 desvio > g/2 |
| R9 | Terceirizada da UI atrasa/entrega abaixo | Média | Médio | Espec L6 pronta antes do contrato; marcos de aceite; CLI já cobre 100% do fluxo sem UI | Sem contrato até S4 |
| R10 | Upstream (sparrow/jagua-rs) muda API ou estagna | Baixa | Baixo | Versões pinadas; MIT/MPL permitem fork; vendorizar se necessário | Breaking change |

---

## 7. Plano de implementação testável

### 7.1 Estratégia de teste (pirâmide)

1. **Unitários** — geometria, gates, conversões (pytest; cobertura nos módulos núcleo);
2. **Contrato** — cada fronteira A→B→C→D→F validada por JSON Schema (`schemas/*.json`) em CI (TD-8 inclui o contrato com spyrrow);
3. **Integração** — TS-1: pasta de fotos padrão → painel validado, semanal e a cada PR;
4. **Benchmark versionado** — TD-7 (nesting) e TB-5 (segmentação): resultados commitados; regressão bloqueia merge;
5. **Validação física** — TA-4 (paquímetro), TF-2 (impressão 1:1), G4 (montagem real). Sem essa camada, o resto é ilusão de progresso.

#### Registro de testes (definição formal de todo ID citado neste doc)

| ID | Definição | Tipo |
|---|---|---|
| TA-1 | Calibração Zhang: RMS < 0,5 px em ≥ 20 poses; resíduos por pose inspecionados | Integração |
| TA-2 | Undistort aplicado a 100% das capturas do ingest (verificação de metadado) | Unitário/contrato |
| TA-3 | Retificação + escala: `residual_check_mm ≤ 1,0` na placa ArUco re-medida | Integração |
| TA-4 | Validação dimensional: 10 peças × paquímetro, erro ≤ 2 mm, espessuras 10–40 mm | **Física** |
| TB-1 | Bake-off de segmentação: IoU > 0,95 (pior classe) e Hausdorff ≤ 2 mm no conjunto de validação | Benchmark |
| TB-2 | Correção interativa: caso difícil resolvido em ≤ 3 prompts | Integração |
| TB-3 | Simplificação: redução > 95% dos pontos, desvio de área < 1%, desvio máx ≤ ε | Unitário |
| TB-4 | Gate de validade: suíte de máscaras patológicas → 0 polígonos inválidos aceitos | Unitário |
| TB-5 | Tempo de segmentação: < 30 s/peça local (MPS) e < 5 s/peça na g5.xlarge (mediana de 20) | Benchmark |
| TC-1 | Proveniência: reconstrução da cadeia painel→run→peças→sessão→perfil→fotos | Integração |
| TC-2 | Ciclo de vida do estoque: `available/reserved/used/rejected` — peça `used` nunca retorna em consulta de montagem | Unitário |
| TC-3 | Backup/restauração: snapshot S3 restaurado íntegro (contagem + hashes) | Integração |
| TD-1 | Strip: sparrow reproduz faixa publicada nas instâncias ESICUP de referência | Benchmark |
| TD-2 | Painel: rota P2 gera solução válida com utilization ≥ baseline BLF e tempo ≤ 10 min | Benchmark |
| TD-3 | Junta: distância mínima ≥ g em 100% dos pares (amostra + validador) | Unitário |
| TD-4 | Restrição de rotação por peça respeitada na solução | Unitário |
| TD-5 | Validador adversarial: suíte com sobreposições/violações conhecidas → 0 falsos aceites | Unitário |
| TD-6 | Determinismo: bit a bit no ambiente de referência pinado; equivalência geométrica (métricas ±0,1%) entre ambientes | CI |
| TD-7 | Benchmark congelado: painel 1 m²/~50 peças reais < 10 min, sem regressão de utilization | Benchmark |
| TD-8 | Contrato com o engine: JSON Schema de entrada/saída do spyrrow/sparrow validado; roda na versão pinada e detecta breaking change em upgrade | Contrato |
| TE-1 | Usabilidade: operador leigo completa capturar→revisar→montar→exportar sem assistência | **Física/UX** |
| TF-1 | Render com textura real gerado para painel de teste | Integração |
| TF-2 | Gabarito 1:1: impressão conferida com régua (±1 mm/m) | **Física** |
| TF-3 | Relatório do painel completo (métricas + proveniência) para 100% dos painéis | Contrato |
| TS-1 | Fumaça end-to-end: pasta de fotos padrão → painel validado (semanal + a cada PR) | Integração |
| TS-2 | Throughput: sessão real ≥ 60 peças/h | **Física** |
| TS-3 | Offline: fluxo completo com rede desabilitada | Integração |
| TS-4 | Custo nuvem: gasto acumulado ≤ pró-rata de 120 h GPU + 500 GB | Monitor |
| TS-5 | Licenças: auditoria automática de dependências compatível com D9 | CI |

Estado incremental em 19/07/2026: o cron semanal executa `petra process-session` sobre uma fixture sintética congelada, cobrindo A→B com ChArUco real, retificação, paralaxe, chroma, pós-processamento e polígono métrico. O resultado é atômico e idempotente por fingerprint/hashes. Essa evidência é registrada como **`TS-1 partial`**; não satisfaz nem declara o TS-1 integral até C–F produzirem um painel validado.

### 7.2 Gates de qualidade

| Gate | Critério objetivo | Quando |
|---|---|---|
| **G1 — Engine confirmado** | sparrow ≥ LBF-baseline + atinge ESICUP publicado; roda instância "tipo retalho" 50 peças < 10 min | Fim de S0 |
| **G2 — Metrologia aprovada** | RNF-01: erro ≤ 2 mm nas 10 peças de referência, 2 sessões independentes | Fim de S1 |
| **G3 — Painel viável** | TD-2: rota P2 produz painel válido com utilization ≥ baseline e tempo ≤ 10 min | Fim de S4 |
| **G4 — Fechamento físico** | Painel-teste interno montado; desvio ≤ g/2 | S5 |
| **G5 — Operável por leigo** | TE-1 com operador real sem assistência | S6–S7 |

### 7.3 Sprints (quinzenais/mensais, ancorados no calendário real)

**S0 — Fundação e prova do engine (20–31/07/2026)**
Entregas: scaffold `petra/` completo; CI (lint+test+licenças, L8/L10); spike sparrow: rodar ESICUP + 50 polígonos sintéticos "tipo retalho"; esboço `calibrate.py`.
Saída testável: **G1** + `cargo run`/`spyrrow` reproduzível documentado + TS-5 verde.

**S1 — Módulo A completo (01–14/08)** *(fecha a entrega contratada CRON-3, M4)*
Entregas: calibração assistida + perfil; retificação ArUco + escala; correção t/Z; `validate_dims.py`.
Saída testável: **G2** (TA-1..4); relatório dimensional versionado (evidência para relatório FAPES).

**S2 — Módulo B: bake-off e pipeline (15/08–14/09)** *(meio de CRON-4; checkpoint 60% do orçamento em 14/09)*
Entregas: conjunto de validação (L3); bake-off D1 (BiRefNet × SAM 3 × SAM 2 × chroma) com relatório; pipeline foto→polígono; auditoria licença SAM 3 (L5); `thickness_mm` por peça (L4).
Saída testável: TB-1..5; decisão **D1 registrada** com dados.

**S3 — Módulo C + strip solver (15/09–14/10)** *(fecha CRON-4; inicia CRON-5)*
Entregas: catálogo completo + ingest + sync S3; RF-11 via spyrrow integrado ao catálogo; protótipo P2 da variante painel; validador independente (RF-15).
Saída testável: TC-1..3, TD-1, TD-5 (suíte adversarial), TD-8; ≥ 200 peças reais catalogadas.

**S4 — Variante painel v1 (15/10–13/11)** *(fecha CRON-5; PC parcial até 13/11)*
Entregas: painel P2 completo (seleção de subconjunto + SA de composição); benchmark congelado TD-7; espec da UI (L6) e contratação do terceiro; **prestação de contas parcial** com evidências dos sprints S0–S4.
Saída testável: **G3**; TD-2/3/4/6; benchmark publicado no repo.

**S5 — Exportação e fechamento físico (15/11–14/12)** *(CRON-6 em curso; 2ª parcela ~14/12)*
Entregas: Módulo F (render + gabarito 1:1 + relatório); **painel-teste físico interno**; decisões D4/D5 com dados; catálogo ≥ 500 peças.
Saída testável: **G4**; TF-1..3.

**S6 — UI integrada e preparação de pilotos (15/12/26–31/01/27)** *(CRON-6/7)*
Entregas: UI (terceiro) integrada via CLI/API; ensaio de piloto interno; materiais de piloto (protocolo L9, painéis-demonstração).
Saída testável: **G5** parcial; TS-1..3 com UI no circuito.

**S7 — Pilotos com marmorarias (fev–mar/27)** *(CRON-7/8)*
Entregas: ≥ 2 pilotos; medição de baseline de desperdício por empresa; ajustes de campo; decisão D8; painéis-demonstração ≥ 3.
Saída testável: métricas de negócio (gestao/11) preenchidas com dados reais; G5 completo.

**S8 — Consolidação, PI e comercial (mar–14/04/27)** *(CRON-8/9/10; fim da vigência 14/04)*
Entregas: validação final e ajustes; registro de software INPI + notificação FAPES; decisão D9 executada; empacotamento comercial; congelamento de release 1.0; insumos do relatório final.
Saída testável: suíte completa verde na release 1.0; matriz §3 com 100% dos testes executados e resultado registrado.

### 7.4 Definição de pronto (global)

Um módulo está pronto quando: contrato da §2 implementado e validado por schema; testes da matriz §3 verdes no CI; benchmark sem regressão; evidência exportada (relatório/curva/foto) arquivada para a prestação de contas; e documentação do doc correspondente atualizada.
