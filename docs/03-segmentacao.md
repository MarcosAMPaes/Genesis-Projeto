# 03 — Segmentação e Extração de Polígonos

Objetivo do estágio: da imagem retificada ao **polígono simples, válido e fiel** de cada fragmento. A qualidade da borda aqui define o encaixe do mosaico lá na frente.

## 3.1 SAM e variantes — escolha do modelo (Decisão D1)

| Modelo | Características | Quando faz sentido |
|---|---|---|
| SAM ViT-H | Melhor qualidade zero-shot; ~2,4 GB; lento sem GPU dedicada | Referência de qualidade (nuvem, lote) |
| SAM ViT-B | ~4× menor; qualidade um degrau abaixo | Compromisso local |
| **SAM 2** | Sucessor (Meta, 2024); mais rápido e preciso; hierarquia Hiera | Candidato padrão — avaliar primeiro |
| MobileSAM | Encoder TinyViT destilado; ordens de magnitude mais leve | Tempo real local/MPS |
| FastSAM | YOLOv8-seg; muito rápido; bordas menos finas | Pré-visualização na UI |
| HQ-SAM | Variante com refinamento de borda | Se a fidelidade de borda das opções acima não bater a meta |

Método de decisão: rodar todos no **conjunto de validação anotado** (≥ 30 fragmentos: granito claro/escuro, mármore com veios, quartzito, superfícies polidas e brutas); escolher o mais leve com IoU > 0,95 e Hausdorff < 2 mm. Registrar a decisão no doc 01.

## 3.2 Estratégia de prompt e cena

- **Fundo controlado resolve metade do problema**: superfície fosca de cor improvável em rocha (verde ou azul fosco). Rocha polida reflete — iluminação difusa (LED 3000–6500K ajustável) e fundo fosco reduzem os dois modos de falha dominantes (reflexo especular e veio confundido com fundo);
- Múltiplos fragmentos por captura: `SamAutomaticMaskGenerator` + filtros (área mínima/máxima, exclusão da região dos marcadores ArUco) **ou** detecção de blobs por cor de fundo → um box prompt por fragmento (mais controlável que o modo automático);
- Fallback interativo na UI: clique do operador = point prompt; é o mecanismo de correção para os < 10% de casos difíceis;
- Batching na nuvem: encoder roda uma vez por imagem; prompts são baratos — agrupar por imagem, não por fragmento.

## 3.3 Pós-processamento da máscara

Pipeline fixo, nesta ordem:

1. Maior componente conexo (remove respingos);
2. Preenchimento de buracos internos (`morphology fill holes`) — buraco real em fragmento é raríssimo; tratar como ruído por padrão;
3. Abertura/fechamento morfológico leve (kernel 3–5 px) só se necessário — morfologia agressiva come borda real;
4. Rejeição automática com log: máscara tocando borda da imagem, área fora da faixa plausível, mais de um componente grande (peças se tocando na mesa → recapturar separadas).

## 3.4 Contorno e simplificação

```python
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contour = max(contours, key=cv2.contourArea)          # ~10⁴ pontos
approx = cv2.approxPolyDP(contour, epsilon, True)      # Douglas-Peucker
```

### Escolha do ε (Douglas-Peucker)

- ε em **unidades físicas**, não pixels: alvo inicial **ε = 0,5 mm**. Conversão correta (o `epsilon` do OpenCV está na unidade das coordenadas do contorno, i.e. pixels): `epsilon_px = ε_mm / scale_mm_px`, com `scale_mm_px` em mm/px — **dividir, não multiplicar**. Assim a fidelidade independe da configuração de captura;
- Douglas-Peucker garante desvio máximo ≤ ε por construção — ε = 0,5 mm mantém o erro de simplificação uma ordem abaixo da tolerância dimensional (2 mm);
- Validar com a curva ε × (nº de pontos, desvio de área, Hausdorff) no conjunto de validação; publicar a curva no relatório de benchmark e fixar o padrão;
- Esperado: ~10⁴ pontos → 100–400 pontos (>95% de redução). Ponto de equilíbrio: menos pontos = NFP mais barato (custo cresce com nº de vértices); pontos demais = nesting lento; de menos = mosaico não fecha.

Alternativa a testar: `shapely.simplify(tolerance, preserve_topology=True)` (mesma família; conveniente pós-conversão).

## 3.5 Validade geométrica (gate de entrada do catálogo)

Todo polígono, antes de entrar no catálogo:

1. `Polygon(pts)` Shapely; se `!is_valid` → `buffer(0)` (conserta autointerseções menores); se ainda inválido ou virar MultiPolygon → rejeitar com log;
2. Orientação normalizada CCW; fechado; sem vértices duplicados consecutivos;
3. Área do polígono ÷ área da máscara ∈ [0,99; 1,01];
4. Conversão px→mm pela escala da sessão; armazenar em mm (doc 05).

## 3.6 Casos difíceis conhecidos (manter registro vivo)

| Caso | Sintoma | Tratamento |
|---|---|---|
| Rocha escura sobre sombra própria | Borda "vaza" para a sombra | Iluminação difusa multi-ângulo; elevar peça em apoio fino; prompt manual |
| Veio da rocha com cor do fundo | Máscara com mordida | Trocar cor do fundo (ter 2 fundos: verde e azul) |
| Reflexo especular em polido | Buraco ou borda serrilhada | Reduzir intensidade, luz difusa; polarizador é upgrade futuro |
| Peças translúcidas (ônix, quartzito fino) | Borda fantasma | Fundo escuro fosco; conferir com backlight desligado |
| Peças se tocando | Um blob para duas peças | Regra operacional: espaçamento mínimo na mesa; rejeição automática detecta |

Cada falha real vira caso no conjunto de validação — o conjunto cresce com a operação.

## 3.7 Metas do estágio (detalhe no doc 06)

IoU > 0,95 · sem intervenção > 90% · < 30 s/fragmento local (MPS) · < 5 s/fragmento (A10G) · redução de pontos > 95% · desvio de área < 1% · Hausdorff < 2 mm.
