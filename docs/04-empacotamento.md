# 04 — Empacotamento Irregular (Nesting) — Doc Central

> **⚠️ Atualização (jul/2026):** a pesquisa de estado da arte no [doc 09](09-estado-da-arte.md) recomenda **não implementar NFP do zero** e adotar `sparrow`/`jagua-rs` (SOTA aberto 2025/26) como motor, mantendo este doc como fundamento teórico e plano de fallback. Ler o doc 09 antes de iniciar o Módulo D.

Você está resolvendo o **2D irregular nesting problem** (família *cutting & packing*, Wäscher et al.): posicionar polígonos simples arbitrários (côncavos, sem eixos privilegiados, todos distintos) numa região, sem sobreposição, otimizando aproveitamento. **NP-difícil**; métodos exatos (MIP) só resolvem instâncias de brinquedo (~10–15 peças). Na prática industrial: geometria exata para viabilidade + heurística construtiva + meta-heurística para busca. É exatamente a arquitetura contratada (NFP + BLF + SA).

## 4.1 Formalização

**Variante principal (painel):** dado painel retangular W×H fixo, junta g, conjunto de peças P com rotações permitidas R — maximizar a área colocada (ou nº de peças ponderado) sem sobreposição e dentro do painel.

**Variante secundária (strip):** largura W fixa, minimizar a altura usada (equivale a "minimizar a altura total do arranjo" da proposta). Mesma engine, função objetivo diferente — implementar as duas.

Distinção relevante do nosso caso vs. nesting clássico de corte (têxtil/chapa): as peças são **únicas e finitas** (estoque de retalhos, sem repetição), a estética importa (distribuição de cor/veio é objetivo secundário futuro), e a junta é parâmetro do produto, não tolerância de processo.

## 4.2 Junta e tolerância — resolvidas por offset, não no solver

Aplicar **buffer de g/2** em cada peça antes do nesting (dilatação); o solver trabalha com as peças "engordadas" tocando-se; o mosaico físico resulta com junta g uniforme. Somar à dilatação a tolerância de corte/posicionamento definida com a marmoraria (Decisão D5).

Implementação: `pyclipper.PyclipperOffset` (JT_MITER com limite, ou JT_ROUND para côncavos agudos) — Clipper trabalha em inteiros: **escalar mm × 1000** (µm) e desescalar na saída. `shapely.buffer` é alternativa; Clipper é mais previsível para offset poligonal puro.

## 4.3 No-Fit Polygon (NFP) — o núcleo geométrico

**Definição:** NFP(A,B) = região dos deslocamentos do ponto de referência de B em que B sobrepõe A. B pode encostar em A exatamente quando seu ponto de referência está na **fronteira** do NFP. Reduz teste de colisão contínuo a ponto-em-polígono.

**Inner-Fit Polygon (IFP):** mesmo conceito contra o **interior** do painel — região onde a peça cabe inteira. Posições viáveis de uma peça = IFP ⊖ união dos NFPs contra as peças já colocadas; candidatos ótimos ficam nos **vértices e arestas dessa fronteira**.

### Métodos de cálculo (em ordem de robustez prática)

1. **Soma de Minkowski via decomposição convexa** — NFP(A,B) = A ⊕ (−B). Convexo⊕convexo é O(n+m) (merge de arestas ordenadas por ângulo). Para côncavos: decompor em partes convexas (triangulação + fusão gulosa, ou algoritmo de Bayazit), calcular NFPs parciais, **unir com Clipper**. Robusto e paralelizável — **recomendado como implementação principal**.
2. **Minkowski do Clipper** — `pyclipper.MinkowskiSum(−B, A)`: pronto, rápido, mas o resultado para côncavos exige pós-processamento cuidadoso (ficar com o anel externo correto e os buracos válidos). Bom para protótipo e para validar o método 1.
3. **Orbiting/sliding (Burke et al. 2007)** — B "desliza" ao redor de A. Elegante, trata côncavos diretamente, mas os casos degenerados (vértice-em-vértice, arestas colineares, encaixes justos) são um campo minado de implementação. É o método do SVGnest. Não recomendo reimplementar do zero.

### Regras de engenharia do NFP

- **Cache agressivo**: NFP depende do par (forma, forma, rotação relativa). Chave: `(hash_geom_A, hash_geom_B, rot_A, rot_B)`. Com peças únicas e r rotações: até C(n,2)·r² entradas — é aqui que a Decisão D3 (nº de rotações) explode ou não o custo. Persistir o cache no catálogo (peças não mudam);
- **Inteiros**: toda a geometria do nesting em µm inteiros (Clipper nativo). Ponto flutuante em geometria de encaixe justo = bugs intermitentes;
- **Validador independente** (invariante de zero sobreposição): checagem final com `shapely.intersection.area < tol` par a par (ou raster fino) — NUNCA confiar só no NFP que gerou a posição;
- Degenerados a testar em suíte própria: polígonos quase colineares, concavidades profundas e estreitas, peças que só encaixam numa orientação, junta zero.

## 4.4 Heurística construtiva — BLF sobre NFP

Ordem de inserção (decidida pela meta-heurística) → para cada peça: calcular região viável (IFP ⊖ ∪NFPs), escolher posição **bottom-left** (menor y, depois menor x) entre os candidatos da fronteira. Alternativas de critério de posição a expor como configuração: bottom-left clássico; menor desperdício local (menor área morta gerada); maior "contato" (comprimento de fronteira compartilhada — bom para mosaico denso).

Baseline obrigatório: **BLF com ordenação por área decrescente** (grandes primeiro, pequenas preenchem vãos). Toda melhoria do SA é medida contra ele.

## 4.5 Simulated Annealing — a busca

**Espaço de busca**: sequência de inserção + rotação de cada peça (+ critério de posição, se exposto). O decodificador (BLF/NFP) transforma solução → layout; a função objetivo avalia o layout.

**Função objetivo (painel):** maximizar `Σ área_colocada` com desempates: − altura do envelope, − dispersão de vãos (compacidade), + bônus de contato. **(strip):** minimizar altura. Manter pesos em arquivo de configuração versionado — a "estética de mosaico" vai ser ajustada com feedback dos pilotos.

**Vizinhanças** (mistura com probabilidades configuráveis): trocar duas peças na sequência; mover peça para posição aleatória da sequência (shift); trocar rotação de uma peça; trocar peça colocada por peça de fora (variante painel — essencial quando nem tudo cabe).

**Cooling**: geométrico, T ← 0,995·T (contratado). T₀ calibrado para aceitar ~80% dos vizinhos piores no início (amostrar ΔE típico e resolver T₀ = ΔE̅/ln(1/0,8)); parada por T mínimo, estagnação (k iterações sem melhora) ou orçamento de tempo. **Sempre guardar a melhor solução já vista** (o SA aceita pioras; o melhor global é o produto).

**Custo**: cada avaliação = um decode BLF completo (caro). Otimizações na ordem certa: cache de NFP (4.3) → decode incremental (reaproveitar prefixo inalterado da sequência) → multi-seed paralelo (n processos com seeds distintas, melhor resultado vence; é a forma certa de usar a GPU/nuvem: paralelismo de corridas, não de geometria).

**Reprodutibilidade**: seed única controla tudo; registro de (seed, params, versão) por corrida no catálogo.

## 4.6 Plano B geométrico — raster híbrido

Se o NFP côncavo travar o cronograma: discretizar peças em bitmaps (grade de 1 mm), colisão = AND bit a bit, BLF varre posições na grade. Perde exatidão (resolução da grade), ganha robustez absoluta e simplicidade. Aceitável para mosaico com junta ≥ 2 mm; inaceitável para encaixe justo. Manter como fallback e como **validador barato** do caminho NFP.

## 4.7 Benchmark

1. **Interno (o que vale):** conjuntos reais de fragmentos do catálogo, congelados e versionados — painel 1 m², ~50 peças. Métricas: aproveitamento %, tempo, nº de peças colocadas, altura (strip). Meta: ≥ 70% de aproveitamento (fixar após 1ª medição), < 10 min no notebook, 0 sobreposições — sempre vs. baseline BLF puro;
2. **Externo (sanidade):** instâncias ESICUP clássicas (shirts, swim, trousers, albano) para comparar a engine com resultados publicados de SA/BLF — detecta bug de qualidade que o conjunto interno não revela.

## 4.8 Extensões futuras (registrar, não implementar agora)

Colocação em buracos/concavidades de peças grandes (NFP com buracos); objetivos estéticos (distribuição de cor/veio/textura — peças têm identidade visual no catálogo); corte leve de peças (relaxar "peça inteira" quando a marmoraria aceitar retrabalho); compactação final por programação linear (Gomes & Oliveira) para "apertar" o layout do SA.
