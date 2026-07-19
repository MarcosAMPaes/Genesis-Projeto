# Corpus de validação

Este diretório é a única exceção autorizada à regra de dados fora do Git. O estado inicial é `draft`: nenhum corpus real foi incluído e L3 permanece aberta.

- `development`: casos usados para ajustar parâmetros; nunca entram no resultado final do bake-off.
- `evaluation`: conjunto congelado de pelo menos 30 amostras usado em TB-1/TB-5.
- Máscaras são PNG binário (`0`/`255`), sem antialiasing e com a mesma resolução da imagem.
- Cada amostra registra backend (`lfs` ou `s3`), tamanho, SHA-256 e atributos de classe no manifesto.
- Antes do primeiro asset LFS, preencher a quota disponível e seu orçamento de 80%; o linter bloqueia blobs LFS sem orçamento.
- Pesos de modelos e dados operacionais não são permitidos neste diretório.
