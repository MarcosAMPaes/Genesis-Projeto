# 09 — Estado da Arte (jul/2026) e Veredito Algorítmico

Análise fundamentada da pergunta: *"os algoritmos escolhidos (Zhang + SAM + Douglas-Peucker + NFP/BLF/SA) ainda são a melhor forma de resolver isso?"* Pesquisa realizada em 19/07/2026 sobre literatura e software de 2024–2026.

## Resumo do veredito

| Estágio | Escolha original | Veredito | Ação |
|---|---|---|---|
| Calibração | Método de Zhang | ✅ **Continua padrão-ouro** | Manter (doc 02) |
| Segmentação | SAM (2023) | ⚠️ **Superado dentro da própria família + concorrente novo** | Bake-off: **SAM 3** × **BiRefNet** × SAM 2 leve |
| Simplificação | Douglas-Peucker | ✅ Adequado | Manter |
| **Empacotamento** | NFP + BLF + SA | ❌ **Não é mais o SOTA (2006–2013). O SOTA aberto de 2025/26 é `sparrow`/`jagua-rs`** | Mudança de arquitetura recomendada (§3) |

A conclusão central: **não implementar NFP do zero**. O módulo mais arriscado do projeto (meses de trabalho em geometria degenerada) foi resolvido pela comunidade em 2024–2025, com código aberto, licença permissiva e desempenho superior ao que nossa arquitetura contratada alcançaria.

---

## 1. Calibração — Zhang permanece correto

Nenhuma mudança de paradigma: Zhang (2000) via OpenCV segue sendo o método padrão para câmera única + plano de referência. Métodos "learned" de calibração existem, mas visam cenários sem padrão de calibração — nosso caso (bancada fixa, tabuleiro disponível) é exatamente o caso ideal do método clássico. **Decisão: manter integralmente o doc 02.**

## 2. Segmentação — dois desenvolvimentos importantes pós-proposta

### SAM 3 (Meta, nov/2025)

O SAM citado na proposta (2023) foi sucedido por SAM 2 (2024) e **SAM 3 (19/11/2025)**, open-source (SAM License), com *Promptable Concept Segmentation*: um prompt de conceito ("fragmento de pedra") segmenta **todas as instâncias de uma vez, com IDs únicos** — elimina a etapa de gerar prompts por peça no nosso caso de múltiplos fragmentos por captura. Meta reporta ~2× de ganho sobre sistemas anteriores em segmentação por conceito, e a variante 3.1 melhora velocidade.

**Implicação:** o fluxo do doc 03 §3.2 (blob detection → box prompt por peça) pode colapsar em um único concept prompt por foto. Manter point-prompt (SAM 2/3 visual) como correção interativa.

### BiRefNet (CAAI AIR 2024) — o concorrente que a proposta não previu

Para o nosso caso específico — **um objeto de primeiro plano sobre fundo controlado, onde a fidelidade fina da borda é o que importa** — a família certa na literatura é *Dichotomous Image Segmentation* (DIS), não segmentação promptable genérica. O SOTA aberto é **BiRefNet** (MIT license): processa 2K/4K sem perder detalhe de borda, é muito mais leve que SAM ViT-H e foi projetado exatamente para extração de contorno fino. Para rodar local no Apple Silicon, é provavelmente a melhor relação fidelidade/custo.

**Decisão D1 reformulada:** bake-off no conjunto de validação anotado com: (a) **BiRefNet** (candidato a padrão local), (b) **SAM 3** concept prompt (candidato a padrão nuvem/lote + correção interativa), (c) SAM 2/MobileSAM (fallback leve), (d) chroma-key clássico como baseline de sanidade (com fundo fosco controlado, o clássico pode ser surpreendentemente competitivo). Critérios: Hausdorff em mm e IoU da pior classe de rocha — não a média.

## 3. Empacotamento — o SOTA mudou em 2025; nossa arquitetura contratada ficou para trás

### O que a pesquisa encontrou

1. **`sparrow`** (Gardeyn, Vanden Berghe & Wauters, KU Leuven — arXiv:2509.13329, set/2025, v3 fev/2026): novo **estado da arte aberto** em 2D irregular strip packing. Abordagem diferente da nossa: decompõe a otimização em uma sequência de **problemas de viabilidade**, movendo itens em colisão até resolvê-las (exploração + compressão), em vez de construir arranjos válidos peça a peça (BLF) e buscar por metaheurística (SA). Os autores reportam superar o SOTA anterior "por margem inesperadamente ampla" nas instâncias ESICUP + 10 instâncias industriais novas. **Rust, MIT, com wrapper Python (`spyrrow` no PyPI), demo WASM.**

2. **`jagua-rs`** (mesmo grupo — INFORMS Journal on Computing 2024): engine de detecção de colisões open-source (MPL-2.0) que separa geometria de otimização. Recursos que nos importam diretamente: **rotação contínua** (não precisamos discretizar rotações!), **distância mínima de separação nativa** (a junta do mosaico vira parâmetro do engine, sem truque de offset), buracos e zonas de qualidade no contêiner, simplificação de polígonos integrada preservando viabilidade, milhões de queries de colisão/segundo. Variantes modeladas: strip, bin, multi-strip (knapsack na lista de pendências).

3. **Linha raster** (Umetani & Murakami, EJOR 2022 — coordinate descent sobre formas rasterizadas): confirma que métodos por resolução de sobreposição/varredura superaram construção+metaheurística clássica. É a mesma família conceitual do sparrow (resolver colisões, não evitá-las).

4. **Métodos exatos (MIP)**: avanços reais (modelos por fatias verticais NFP-CM-VS, EJOR 2024; NDTM, 2024), mas seguem restritos a instâncias pequenas — não servem ao nosso caso operacional (~50 peças, minutos de orçamento), no máximo para otimalidade em painéis pequenos.

5. **Deep RL / learning-based nesting** (2023–2026): área ativa (frameworks híbridos DRL+heurística, GNNs, "dense reward"), com ganhos reportados contra baselines fracos. **Nenhum software robusto, reproduzível e superior às heurísticas maduras está disponível.** Não é opção de produção em 2026; é tema de acompanhamento.

### O que isso significa para nós

A arquitetura contratada (NFP+BLF+SA) descreve o estado da arte de ~2006–2013. Ela funcionaria — mas construiríamos, com esforço alto e risco alto (NFP robusto para côncavos é notoriamente traiçoeiro), um motor **pior** do que um que já existe pronto, aberto e superior.

**Arquitetura recomendada (v2):**

| Camada | Ferramenta | Justificativa |
|---|---|---|
| Detecção de colisão + junta + rotação contínua | **`jagua-rs`** (via Rust ou binding) | Elimina o módulo mais arriscado (NFP próprio); junta nativa via `min_item_separation` |
| Variante strip (altura mínima) | **`sparrow`** (via `spyrrow`) | SOTA de graça; resolve a variante contratada "minimizar altura total" |
| Variante painel (mosaico: subconjunto de peças em painel fixo) | **Camada própria** de seleção de subconjunto + busca (SA) orquestrando o engine | É a nossa variante principal e o sparrow não a cobre nativamente (knapsack é "todo" no jagua-rs). Aqui está nossa contribuição real |
| Objetivo estético (cor/veio/distribuição) | Camada própria (função objetivo da busca) | Inexistente na literatura de nesting — nosso diferencial de produto |
| Validador independente | Shapely/raster próprio | Invariante de zero sobreposição continua conosco |

**Onde fica o valor proprietário do projeto:** não no motor geométrico (commodity aberta em 2026), e sim em (i) o pipeline metrológico foto→polígono confiável, (ii) a variante painel/mosaico com seleção de subconjunto de peças únicas de estoque, (iii) o objetivo estético, e (iv) o fechamento físico (gabarito → montagem). É exatamente onde nenhuma ferramenta existente chega.

### Compatibilidade com o compromisso da proposta

A proposta FAPES descreve BLF, NFP e SA. A recomendação **preserva os conceitos**: o jagua-rs nasce da literatura de NFP/colisão (o `lbf` de referência do próprio projeto é um Bottom-Left-Fill), e nossa camada de busca para a variante painel continua sendo **Simulated Annealing**. Trocar implementação interna por um engine superior, mantendo o objeto (sistema de otimização de arranjo de fragmentos), é evolução técnica normal de P&D — registrar no relatório técnico como aprimoramento fundamentado na literatura 2024–2026, citando as fontes deste doc. Mudança material de escopo não há; ainda assim, mencionar a evolução na prestação de contas parcial (evidencia domínio do estado da arte — pesa a favor).

### Plano de validação (antes de casar com a decisão)

1. **Semana 1 do Módulo D:** rodar `sparrow` (binário pronto) nas instâncias ESICUP e num conjunto sintético de ~50 polígonos "tipo retalho" (côncavos, únicos) — medir qualidade e tempo no notebook;
2. Prototipar a variante painel: (a) via strip + corte na altura do painel, (b) via camada própria de subconjunto sobre o engine — comparar;
3. Só se ambos falharem nos requisitos (improvável): retornar ao plano NFP próprio do doc 04, que permanece documentado como fallback;
4. Benchmark head-to-head registrado — a decisão final é por dados, não por paper.

### Riscos da adoção (honestidade)

| Risco | Mitigação |
|---|---|
| Stack Rust (equipe é Python) | `spyrrow` (PyPI) para strip; para a camada painel, interop via subprocess/JSON do sparrow ou binding fino; Rust fica encapsulado |
| Variante painel exige trabalho próprio sobre o engine | É o trabalho que faríamos de qualquer forma — só que sobre base sólida em vez de NFP artesanal |
| Dependência de projeto de terceiros | MIT/MPL-2.0 permitem fork; código na nossa órbita desde o dia 1; jagua-rs publicado em periódico (IJOC) com CI e benchmarks públicos |
| Licenças | MIT (sparrow) e MPL-2.0 (jagua-rs) são compatíveis com nosso GPL-3.0 e com uso comercial |

## 4. Impacto nos demais docs

- **doc 03**: D1 reformulada (§2 acima) — incluir BiRefNet e SAM 3 no bake-off;
- **doc 04**: permanece válido como fundamento teórico e plano de fallback; a arquitetura v2 deste doc tem precedência prática;
- **doc 07**: Módulo D reordenado — começa pela avaliação do sparrow/jagua-rs (1 semana), não pela implementação de NFP;
- **doc 08**: referências novas adicionadas.

## Fontes primárias desta análise

- Gardeyn, Vanden Berghe, Wauters — *An open-source heuristic to reboot 2D nesting research* (sparrow). [arXiv:2509.13329](https://arxiv.org/abs/2509.13329) · [github.com/JeroenGar/sparrow](https://github.com/JeroenGar/sparrow) (MIT)
- Gardeyn et al. — *Decoupling Geometry from Optimization in 2D Irregular C&P Problems* (jagua-rs). INFORMS Journal on Computing, 2024. [doi:10.1287/ijoc.2024.1025](https://doi.org/10.1287/ijoc.2024.1025) · [github.com/JeroenGar/jagua-rs](https://github.com/JeroenGar/jagua-rs) (MPL-2.0)
- [`spyrrow`](https://pypi.org/project/spyrrow/) — wrapper Python do sparrow (PyPI)
- Umetani & Murakami — *Coordinate descent heuristics for the irregular strip packing problem of rasterized shapes*. [EJOR 303(3), 2022](https://www.sciencedirect.com/science/article/pii/S0377221722002582) · [arXiv:2104.04525](https://arxiv.org/abs/2104.04525)
- MIP recente: [NFP-CM-VS, arXiv:2206.00032 / EJOR 2024](https://arxiv.org/abs/2206.00032); [NDTM, Mathematics 12(15), 2024](https://doi.org/10.3390/math12152414)
- DRL em nesting (panorama): [J. Intelligent Manufacturing, 2025](https://link.springer.com/article/10.1007/s10845-025-02620-6); [IJSC 2024](https://www.worldscientific.com/doi/10.1142/S1793351X24430025); [arXiv:2309.10329](https://arxiv.org/pdf/2309.10329)
- Meta AI — *SAM 3: Segment Anything with Concepts* (19/11/2025). [ai.meta.com](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/) · [blog SAM 3.1](https://ai.meta.com/blog/segment-anything-model-3/) · [análise Roboflow](https://blog.roboflow.com/what-is-sam3/)
- Zheng et al. — *Bilateral Reference for High-Resolution Dichotomous Image Segmentation* (BiRefNet). CAAI AIR 2024. [github.com/ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet) (MIT)
