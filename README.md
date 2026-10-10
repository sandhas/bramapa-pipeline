# bramapa-pipeline

Pipeline offline do **Bramapa**, um atlas 2.5D isométrico em pixel art do Brasil. Este repo é a
parte em Python que pega geodados (relevo, uso do solo, malhas do IBGE...) e cospe os arquivos que
o motor renderiza: chunks de tiles (`.bmc`) e o manifesto de cada nível (`level.toml`).

**Estado atual: Fase 1.** O esqueleto funciona de ponta a ponta (codec do `.bmc`, escrita de
`level.toml`, CLI, testes) e os ids de tile já saem do `tiles.toml` de um tileset via autotiling,
mas o terreno ainda é sintético. Nada de geodado de verdade entra aqui
ainda; isso é a Fase 2.

## Os quatro repositórios

- `bramapa` (motor, Rust + wgpu) consome `.bmc` + `level.toml` + tileset e desenha;
  `bramapa-tileset` publica o atlas de sprites e o `tiles.toml` que dá nome aos ids de tile.
- `bramapa-pipeline` (este) gera os chunks a partir de geodados; `bramapa-data` cuida dos addons de
  dados tabulares e séries temporais que o motor sobrepõe ao mapa.

Os contratos entre eles estão em `bramapa/docs/spec/`. Este repo implementa **`spec_version = 0`**
de [`chunk.md`](../bramapa/docs/spec/chunk.md) (formato `.bmc` e `level.toml`) e referencia os ids
de tile de [`tileset.md`](../bramapa/docs/spec/tileset.md). A v0 é instável por definição: quando a
spec mudar, isto aqui muda junto, sem compatibilidade retroativa.

## Setup

Precisa de [`uv`](https://docs.astral.sh/uv/) e Python >= 3.12.

```sh
uv sync                       # cria .venv e instala tudo (numpy, zstandard, typer; pytest no grupo dev)
uv run pytest                 # testes
uv run bramapa-pipeline --help
```

### Gerar um nível de teste

```sh
uv run bramapa-pipeline chunk-test --out ../bramapa-data-test
uv run bramapa-pipeline chunk-test --out ../bramapa-data-test --width 512 --height 512 --seed 7
uv run bramapa-pipeline chunk-test --out ../bramapa-data-test --tileset ../bramapa-tileset
```

Escreve `level.toml` (nível 1, `pais:BR`, "Teste sintético", `cell_meters = 8000`, `EPSG:5880`,
tileset `bramapa-base@0.0.1` ou o id/versão do `--tileset` dado, sem período) e um `chunks/<cx>_<cy>.bmc` para cada chunk de 64×64.
O terreno vem de value noise em numpy: um continente "blobby" cercado de oceano, com areia na costa,
lagos, campo, floresta, savana e montanha; um trecho urbano pequeno; duas rodovias e um rio no
overlay; árvores na floresta e prédios/casas na cidade como objetos. Com a mesma semente o resultado
é idêntico byte a byte.

### Inspecionar

```sh
uv run bramapa-pipeline inspect ../bramapa-data-test                  # tabela com todos os chunks
uv run bramapa-pipeline inspect ../bramapa-data-test/chunks/1_1.bmc   # cabeçalho + stats de um chunk
uv run bramapa-pipeline inspect ../bramapa-data-test --tileset ../bramapa-tileset  # nomeia as variantes
```

## Autotiling

O gerador não conhece ids de tile. Ele produz grades de **cobertura** (`"oceano"`, `"campo"`,
`"agua"`... no terreno; `"rio"`, `"rodovia"`... no overlay) e objetos por **nome** (`"arvore"`);
`chunkify()` resolve tudo para ids através de um `Tileset` (`tileset.py`), carregado do
`tiles.toml` de um tileset ou, sem `--tileset`, do tileset embutido que reproduz a tabela de ids
provisórios abaixo (um tile por cobertura, sem regras). É um caminho de código só.

Para cada célula o pipeline calcula a **máscara de vizinhança** de 4 bits do `tileset.md`:

| Bit | Vizinho | Coordenada |
|---|---|---|
| `N = 1` | norte | `(x, y−1)` |
| `E = 2` | leste | `(x+1, y)` |
| `S = 4` | sul | `(x, y+1)` |
| `W = 8` | oeste | `(x−1, y)` |

O bit fica ligado quando o vizinho tem a **mesma cobertura**. Fora da grade conta como "mesma
cobertura", então a borda do mapa não ganha contorno (uma célula de água encostada na borda norte
e cercada de campo tem máscara `1`, não `0`). O tile é `[[autotile]].masks[máscara]` da cobertura,
ou `default` quando a máscara não está listada; coberturas sem regra precisam ter exatamente um
tile da classe certa (`terrain` ou `overlay`) com aquela `cover`, senão é erro. Célula vazia é `0`.

Tudo isso acontece aqui, em numpy vetorizado (`autotile.py`), sobre a grade inteira antes de
fatiar em chunks. **O motor nunca calcula máscara**: ele só vê ids de tile e desenha a cell do
atlas correspondente. Com o placeholder `bramapa-base`, uma margem de lago sai como `100 + máscara`
e uma costa como `120 + máscara`.

```sh
uv run bramapa-pipeline autotile-demo --tileset ../bramapa-tileset   # lago ASCII: máscara e id por célula
```

## Ids de tile provisórios

Sem `--tileset`, o tileset embutido (`tileset.builtin_tileset()`, construído a partir de
`src/bramapa_pipeline/tiles.py`) usa esta convenção, a mesma do `bramapa-base@0.0.1`. `0` é sempre
vazio.

| Camada | Ids |
|---|---|
| terrain | 1 campo, 2 agua, 3 oceano, 4 floresta, 5 areia, 6 urbano, 7 montanha, 8 savana, 9 agricultura, 10 pastagem, 11 caatinga, 12 pantanal |
| overlay | 20 rodovia, 21 rio, 22 limite_estado, 23 limite_municipio, 24 limite_pais, 25 ferrovia |
| objects | 40 arvore, 41 predio, 42 casa |

## Layout do código

```
src/bramapa_pipeline/
  chunk.py   # Chunk, encode/decode, write_chunk/read_chunk — o .bmc byte a byte
  level.py    # LevelManifest, write_level_toml/read_level_toml
  tileset.py  # Tileset/TileDef/AutotileRule, load_tileset(tiles.toml), builtin_tileset()
  autotile.py # máscara de vizinhança vetorizada e apply_autotile(coberturas -> ids)
  tiles.py    # ids provisórios acima (só alimentam o tileset embutido)
  synth.py    # gerador sintético (value noise) em coberturas e chunkify() via tileset
  cli.py      # typer: chunk-test, inspect, autotile-demo, version
tests/        # pytest: round-trip, cabeçalho, máscaras, tileset, CLI (fixture em tests/fixtures/)
```

Identificadores e comentários no código em inglês; documentação em PT-BR.

## Próximos passos

- **Fase 2 — nível 1 (país) com dados reais.** Natural Earth para costa/relevo grosso, malhas do
  IBGE para limites de país/estado/município (`limite_*` no overlay), MapBiomas para a classificação
  de cobertura (floresta, savana, caatinga, campo, pantanal, agricultura, pastagem, urbano...).
  Entram `rasterio` e `geopandas` como dependências; a grade é amostrada em `EPSG:5880` com eixos
  rotacionados 45° como manda o `README.md` da spec.
- **Autotiling de overlay por máscara.** Rodovias, rios e ferrovias já passam pelo mesmo caminho
  de `[[autotile]]`, mas o placeholder ainda desenha uma faixa reta; curvas e ramificações entram
  quando o tileset tiver as variantes.
- **Hot paths em Rust.** Reamostragem e classificação em escala de município (1024² células × milhares
  de municípios) vão ficar lentas em Python puro; a ideia é mover esses trechos para uma extensão
  em Rust (PyO3) mantendo a orquestração em Python.

## Licença

Dupla licença MIT ou Apache-2.0, à sua escolha. Veja `LICENSE-MIT` e `LICENSE-APACHE`.
