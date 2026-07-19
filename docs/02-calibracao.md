# 02 — Calibração e Metrologia da Captura

Objetivo do estágio: transformar uma foto em uma **medição**. Saída: imagem fronto-paralela com escala mm/px conhecida, erro dimensional ≤ 2 mm (alvo 1 mm).

## 2.1 Método de Zhang (calibração intrínseca)

Zhang (2000): a câmera observa um plano com padrão conhecido (tabuleiro de xadrez) em várias poses; cada pose dá uma homografia; o conjunto restringe os parâmetros intrínsecos. Solução fechada + refinamento não linear (Levenberg-Marquardt) minimizando erro de reprojeção.

Parâmetros estimados:

- **K** (matriz intrínseca): distâncias focais fx, fy; centro óptico cx, cy
- **Distorção**: radial k1, k2, k3; tangencial p1, p2 (modelo padrão OpenCV de 5 coeficientes)
- Extrínsecos por pose (rvec, tvec) — úteis para diagnóstico, não para produção

### Implementação OpenCV

```python
# 1. Detecção — usar a variante SB, mais robusta a iluminação/ângulo
ok, corners = cv2.findChessboardCornersSB(gray, (cols, rows), flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
# (fallback clássico: findChessboardCorners + cornerSubPix com critério de término apertado)

# 2. Calibração
rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, gray.shape[::-1], None, None)

# 3. Correção (mapa pré-computado uma vez por perfil, remap por imagem)
map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K_new, size, cv2.CV_16SC2)
undistorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
```

### Protocolo de captura da calibração

1. ≥ 20 poses do tabuleiro; inclinações variadas (±30–45°) e posições cobrindo **todo o campo de visão, inclusive cantos** (é onde a distorção radial é maior);
2. Tabuleiro plano e rígido (impresso colado em superfície plana — empenamento arruína a calibração); quadrados de dimensão certificada (kit orçado);
3. Foco e zoom **travados** durante toda a sessão — mudou foco, recalibrou;
4. Aceite: RMS < 0,5 px; inspecionar resíduos por imagem (uma pose ruim contamina o conjunto — descartar outliers e recalibrar).

## 2.2 Especificidades do iPhone 15 Pro

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
