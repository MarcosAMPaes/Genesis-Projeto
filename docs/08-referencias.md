# 08 — Referências Técnicas

Fontes canônicas por tema. Prioridade de leitura marcada com ★ (essencial para implementar).

## Calibração e metrologia

- ★ Zhang, Z. — *A Flexible New Technique for Camera Calibration*. IEEE TPAMI, 2000. (O método contratado; leitura curta e direta)
- Documentação OpenCV: *Camera Calibration and 3D Reconstruction* (módulo `calib3d`) e módulo `aruco` (ChArUco/ArUco para retificação e escala)
- Hartley & Zisserman — *Multiple View Geometry in Computer Vision*, 2ª ed. (referência de fundo: homografias, cap. 2 e 4)

## Segmentação

- ★ Kirillov et al. — *Segment Anything* (SAM). Meta AI, 2023. arXiv:2304.02643
- Ravi et al. — *SAM 2: Segment Anything in Images and Videos*. Meta AI, 2024. arXiv:2408.00714
- Zhang et al. — *Faster Segment Anything: Towards Lightweight SAM for Mobile Applications* (MobileSAM), 2023. arXiv:2306.14289
- Zhao et al. — *Fast Segment Anything* (FastSAM), 2023. arXiv:2306.12156
- Ke et al. — *Segment Anything in High Quality* (HQ-SAM), NeurIPS 2023. arXiv:2306.01567

## Simplificação de contornos

- ★ Douglas, D. & Peucker, T. — *Algorithms for the reduction of the number of points required to represent a digitized line or its caricature*. Cartographica, 1973. (Ramer, 1972, é o precursor independente)

## Empacotamento irregular (nesting) — núcleo do projeto

- ★ Bennell, J. & Oliveira, J.F. — *The geometry of nesting problems: A tutorial*. European Journal of Operational Research, 2008. (O melhor ponto de entrada para NFP/IFP; implemente com este ao lado)
- ★ Burke, E., Hellier, R., Kendall, G., Whitwell, G. — *Complete and robust no-fit polygon generation for the irregular stock cutting problem*. EJOR, 2007. (Método orbital; ler pelos casos degenerados, mesmo sem implementá-lo)
- Burke, E. et al. — *A new bottom-left-fill heuristic algorithm for the two-dimensional irregular packing problem*. Operations Research, 2006.
- Gomes, A.M. & Oliveira, J.F. — *Solving irregular strip packing problems by hybridising simulated annealing and linear programming*. EJOR, 2006. (SA para nesting + compactação por LP — o "apertar" futuro)
- Albano, A. & Sapuppo, G. — *Optimal allocation of two-dimensional irregular shapes using heuristic search methods*. IEEE Trans. SMC, 1980. (Clássico fundador)
- Kirkpatrick, S., Gelatt, C., Vecchi, M. — *Optimization by Simulated Annealing*. Science, 1983.
- Wäscher, G., Haußner, H., Schumann, H. — *An improved typology of cutting and packing problems*. EJOR, 2007. (Taxonomia — situa a variante exata do nosso problema)
- **ESICUP** (EURO Special Interest Group on Cutting and Packing) — instâncias de benchmark públicas (shirts, swim, trousers, albano etc.)

## Implementações de referência (ler o código)

- **SVGnest** (github.com/Jack000/SVGnest) — nesting irregular em JS: NFP orbital + GA. Melhor referência de "como tudo se conecta"; nossa busca é SA, não GA
- **libnest2d** (github.com/tamasmeszaros/libnest2d) — C++ usado pelo PrusaSlicer: NFP + heurísticas; arquitetura de cache e placement
- **Clipper2** (github.com/AngusJohnson/Clipper2) / **pyclipper** — boolean ops, offset e Minkowski em inteiros; base da nossa geometria
- **Shapely 2.x** (shapely.readthedocs.io) — geometria em Python (GEOS): validação, buffer, interseção, Hausdorff
- **segment-anything / sam2** (github.com/facebookresearch) — implementações oficiais dos modelos

## Prática de uso

1. Antes de codar o Módulo D: ler Bennell & Oliveira (2008) inteiro + skim do SVGnest;
2. Toda técnica adotada de um paper ganha comentário no código com a citação (rastreabilidade de decisões);
3. Este arquivo cresce: nova fonte usada → entra aqui com uma linha dizendo para quê.
