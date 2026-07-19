# Petra Smart — Documentação Técnica

Norte do desenvolvimento de software. Gestão do projeto (edital FAPES, cronograma contratual, orçamento, compliance) fica em [`/gestao`](../gestao/README.md) — nada disso aqui.

## O problema, em uma linha

**Empacotamento 2D de polígonos altamente irregulares** (*irregular nesting/strip packing*, NP-difícil): dado um conjunto de fragmentos de rocha com contornos arbitrários — côncavos, sem lados retos, todos únicos — dispor o máximo deles num painel, sem sobreposição, com junta controlada e mínimo espaço morto. Todo o resto do pipeline (calibração, segmentação, catálogo) existe para alimentar esse problema com polígonos **dimensionalmente confiáveis** e depois materializar a solução num gabarito executável.

## Índice

| Doc | Conteúdo |
|---|---|
| [01-arquitetura.md](01-arquitetura.md) | Pipeline completo, estágios, infraestrutura, decisões em aberto |
| [02-calibracao.md](02-calibracao.md) | Método de Zhang, protocolo de captura, retificação, escala px→mm, fontes de erro |
| [03-segmentacao.md](03-segmentacao.md) | SAM e variantes, pós-processamento de máscara, contorno, Douglas-Peucker, validade geométrica |
| [04-empacotamento.md](04-empacotamento.md) | **Doc central**: formalização do nesting, NFP/IFP, BLF, Simulated Annealing, junta, bibliotecas |
| [05-dados-catalogo.md](05-dados-catalogo.md) | Modelo de dados, unidades, formatos, esquema SQLite, layout S3, reprodutibilidade |
| [06-metricas-tecnicas.md](06-metricas-tecnicas.md) | Metas numéricas e protocolos de medição por estágio |
| [07-roadmap.md](07-roadmap.md) | Módulos A–F, ordem de desenvolvimento, critérios de aceite |
| [08-referencias.md](08-referencias.md) | Papers e ferramentas canônicas por tema |
| [09-estado-da-arte.md](09-estado-da-arte.md) | 🔍 **Pesquisa jul/2026 + veredito**: sparrow/jagua-rs, SAM 3, BiRefNet — arquitetura v2 |
| [10-especificacao-operacional.md](10-especificacao-operacional.md) | 📌 **FONTE DE VERDADE da implementação**: contratos A–F, matriz de requisitos, contradições, decisões, riscos, plano testável |

## Princípios de engenharia do projeto

1. **Milímetros, não pixels.** Toda geometria a jusante da calibração vive em unidades físicas (mm). Pixel é detalhe de implementação dos estágios 1–3.
2. **Zero sobreposição é invariante, não métrica.** Solução com colisão é inválida por definição; validador independente do algoritmo de posicionamento.
3. **Reprodutibilidade.** Toda execução de empacotamento registra seed, parâmetros e versão do código; mesmo input → mesmo output.
4. **Benchmark versionado.** Cada release roda a suíte padrão; regressão de qualidade ou tempo bloqueia merge.
5. **O gabarito é o produto.** O sistema só está completo quando o arranjo digital vira mosaico físico que fecha na montagem — junta e tolerância de corte fazem parte do modelo, não são detalhe.

## Stack

Python 3.11+ · OpenCV · PyTorch (SAM; MPS local / CUDA na nuvem) · Shapely 2.x + pyclipper/Clipper2 (geometria) · SQLite (catálogo) · AWS g5.xlarge + S3 (lotes e backup).
