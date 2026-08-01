# 02 — Calibração e Metrologia da Captura

Objetivo do estágio: transformar uma foto em uma **medição**. Saída: imagem fronto-paralela com escala mm/px conhecida, erro dimensional ≤ 2 mm (alvo 1 mm).

## 2.0 Padrão físico único — placa ChArUco

O projeto usa **uma única placa física** nas duas etapas (intrínseca e retificação de sessão): placa plástica adesivada 30×40 cm com ChArUco gerado por `cv2.aruco.CharucoBoard`.

| Parâmetro | Valor | Config |
|---|---|---|
| Quadrados | 7 × 9 | `squares_x`, `squares_y` |
| Lado do quadrado | **38,0 mm** (nominal) | `square_length_mm` |
| Lado do marcador | **28,0 mm** (nominal) | `marker_length_mm` |
| Dicionário | `DICT_5X5_100` (IDs 0–30) | `dictionary` |
| Cantos internos | 48 | derivado |
| Área impressa | 266 × 342 mm | derivado |

Config: [`config/boards/charuco-a3-7x9-38mm.json`](../config/boards/charuco-a3-7x9-38mm.json) — mesmo arquivo para `calibrate create`, `calibrate rectify` e `process-session`.

> **Obrigatório antes da primeira calibração:** medir com paquímetro o lado real de um quadrado impresso (média de 3 quadrados distantes entre si) e, se divergir do nominal, corrigir `square_length_mm` e `marker_length_mm` no JSON. A arte é A3 (297×420 mm) e a placa é 300×400 mm — qualquer reescala da gráfica propaga **erro de escala direto para as medições em mm**. Registrar a medição no dossiê da bancada.

Por que ChArUco também na intrínseca: os cantos de tabuleiro dão precisão subpixel, os marcadores ArUco dão identidade a cada canto — logo, **poses parciais são utilizáveis** (o padrão pode sair pela borda do quadro), o que é essencial para cobrir cantos do campo de visão, onde a distorção radial é maior. Recomendação do próprio OpenCV para medição de alta precisão.

## 2.1 Método de Zhang (calibração intrínseca sobre ChArUco)

Zhang (2000): a câmera observa um plano com padrão conhecido em várias poses; cada pose dá uma homografia; o conjunto restringe os parâmetros intrínsecos. Solução fechada + refinamento não linear (Levenberg-Marquardt) minimizando erro de reprojeção.

Parâmetros estimados:

- **K** (matriz intrínseca): distâncias focais fx, fy; centro óptico cx, cy
- **Distorção**: radial k1, k2, k3; tangencial p1, p2 (modelo padrão OpenCV de 5 coeficientes)
- Extrínsecos por pose (rvec, tvec) — úteis para diagnóstico, não para produção

### Implementação OpenCV

```python
# 1. Detecção do tabuleiro ChArUco (cantos subpixel + IDs)
detector = cv2.aruco.CharucoDetector(board)
charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

# 2. Correspondência 3D↔2D da vista (aceita vista parcial)
object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)

# 3. Calibração com listas por pose (contagem de pontos variável)
rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None,
                                                 flags=cv2.CALIB_FIX_K4|cv2.CALIB_FIX_K5|cv2.CALIB_FIX_K6)

# 4. Correção (mapa pré-computado uma vez por perfil, remap por imagem)
map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K_new, size, cv2.CV_16SC2)
undistorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
```

### Protocolo de captura da calibração

1. ≥ 20 poses válidas; inclinações variadas (±30–45°) e posições cobrindo **todo o campo de visão, inclusive cantos** (é onde a distorção radial é maior) — vistas parciais são bem-vindas e contam, desde que exponham ≥ 12 cantos;
2. Placa plana e rígida — a placa adesivada resolve isso; empenamento arruína a calibração;
3. Foco e zoom **travados** durante toda a sessão — mudou foco, recalibrou;
4. Poses degeneradas (cantos colineares) são descartadas automaticamente pelo detector;
5. Aceite: RMS < 0,5 px; inspecionar resíduos por pose (uma pose ruim contamina o conjunto — excluir explicitamente via `--exclude` e recalibrar; nunca há remoção silenciosa).

## 2.2 Especificidades do iPhone 17 Pro (aparelho real do projeto)

- Usar **sempre a lente principal 1×** (48 MP). Nunca deixar o iOS trocar de lente automaticamente (macro/0.5×/2× têm intrínsecos diferentes — cada lente é outra câmera);
- Travar foco e exposição por sessão (captura via app com controle manual AVFoundation ou app de câmera manual); AF contínuo muda a distância focal efetiva;
- Desativar HDR/Deep Fusion se introduzirem inconsistência de bordas; preferir formato de máxima resolução com processamento mínimo;
- **LiDAR**: papel de verificação — medir distância câmera→plano da bancada por sessão e comparar com a distância implícita da calibração; divergência = alerta de que algo mudou fisicamente;
- Smartphone fixo no trilho da bancada; disparo remoto ou timer (toque na tela move o aparelho).

## 2.3 Retificação fronto-paralela e escala px→mm

Mesmo com câmera "perpendicular", há inclinação residual. Não confiar na montagem; corrigir por software:

1. Colocar na cena um **padrão de referência plano no nível da face superior dos fragmentos** — recomendação: placa com 4+ marcadores **ArUco/ChArUco** em posições conhecidas (OpenCV `cv2.aruco`). ArUco dá detecção automática, identidade de cada canto e robustez que um objeto de referência genérico não dá;
2. Homografia H dos 4+ cantos detectados → retificação (`cv2.warpPerspective`) para vista fronto-paralela;
3. Escala mm/px calculada da geometria conhecida do padrão **após** retificação; registrar no metadado da sessão;
4. Validação contínua: o próprio padrão re-medido na imagem retificada deve bater com as dimensões nominais dentro da tolerância.

### Erro de paralaxe por espessura (fonte de erro dominante)

O contorno visível de um fragmento com espessura *t* está mais próximo da câmera que o plano de referência da mesa. Com distância câmera–plano *Z*, a escala aparente difere por fator ≈ Z/(Z−t): superestimação de ~t/Z.

- Exemplo: Z = 800 mm, t = 20 mm → erro ~2,5% → **12,5 mm em uma peça de 500 mm**. Inaceitável.
- Mitigações (combinar): referência ArUco **na altura da face superior** das peças (elimina o grosso do erro para espessura uniforme); registrar espessura da chapa por sessão e corrigir escala por t/Z (Z via LiDAR); maximizar Z dentro do que a resolução permitir.
- Peças de espessuras variadas na mesma sessão → agrupar por espessura ou corrigir por peça.

## 2.4 Outras fontes de erro (orçamento de erro)

| Fonte | Ordem de grandeza | Controle |
|---|---|---|
| Distorção radial residual | < 0,5 px após undistort | RMS da calibração; recalibrar |
| Inclinação residual da câmera | corrigida pela homografia | Retificação por sessão, não por montagem |
| Paralaxe por espessura | ~t/Z (dominante se ignorada) | § 2.3 |
| Sombra/penumbra na borda | 1–3 px de ambiguidade | Iluminação difusa; critério de borda consistente (doc 03) |
| Empenamento do padrão de referência | proporcional ao empeno | Placa rígida certificada |
| Resolução espacial | mm/px da configuração | Com 48 MP a ~80 cm cobrindo ~60 cm de mesa: ≈0,08 mm/px — folga ampla; a precisão é limitada pelos itens acima, não pela resolução |

## 2.5 Perfil de calibração (artefato versionado)

`calib_profile.json`: K, dist, RMS, data, resolução, id do aparelho/lente, altura Z (LiDAR), escala mm/px, hash das imagens de calibração, id da configuração física da bancada. Toda captura referencia o perfil usado (rastreabilidade no catálogo — doc 05). Recalibrar quando: mudança física da bancada, troca/atualização do aparelho, divergência LiDAR×calibração, ou validação contínua fora da tolerância.

## 2.6 Validação dimensional (protocolo do doc 06)

10 fragmentos de referência medidos com paquímetro (2+ eixos cada) × medição do sistema; relatório com erro médio, máximo e por eixo; rodar a cada perfil novo e semanalmente como teste de fumaça.
