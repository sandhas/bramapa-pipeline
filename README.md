# bramapa-pipeline

Pipeline offline do **Bramapa**, um atlas 2.5D isométrico em pixel art do Brasil. Este repo é a
parte em Python que pega geodados (relevo, uso do solo, malhas do IBGE...) e cospe os arquivos que
o motor renderiza: chunks de tiles (`.bmc`) e o manifesto de cada nível (`level.toml`).

**Estado atual: Fase 2.** O esqueleto funciona de ponta a ponta (codec do `.bmc` com as camadas
`terrain`, `overlay`, `entity` e `objects`, escrita de `level.toml`, CLI, testes), os ids de tile
saem do `tiles.toml` de um tileset via autotiling e o **nível 1 (Brasil e vizinhos) já é gerado de
geodados públicos** (Natural Earth, IBGE, ETOPO) pelo comando `build-level1`. O gerador sintético
(`chunk-test`) continua existindo para testar o motor sem download.

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
uv sync                       # cria .venv e instala tudo (numpy, zstandard, typer, rasterio, geopandas...; pytest no grupo dev)
uv run pytest                 # testes
uv run bramapa-pipeline --help
```

As dependências geoespaciais (`rasterio`, `geopandas`, `pyogrio`, `shapely`, `pyproj`) vêm como
wheels binárias com GDAL/PROJ embutidos; não precisa instalar nada no sistema. Há wheels para o
Python 3.14 (o `.python-version` do repo).

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

## Nível 1 real

`build-level1` monta o nível 1 (`pais:BR`, "Brasil e vizinhos") a partir de geodados públicos,
sem login. Primeiro o download para o cache, depois a construção:

```sh
uv run bramapa-pipeline fetch                                   # baixa tudo para data/raw/ (~70 MB)
uv run bramapa-pipeline build-level1 --out ../bramapa-data/1/pais_BR --tileset ../bramapa-tileset
uv run bramapa-pipeline build-level1 --out /tmp/l1 --tileset ../bramapa-tileset --preview /tmp/l1.png
uv run bramapa-pipeline inspect ../bramapa-data/1/pais_BR --tileset ../bramapa-tileset
```

Opções de `build-level1`: `--cell-meters 7000`, `--size 1024`, `--center-lon -53`,
`--center-lat -14`, `--cache-dir data/raw`, `--preview arquivo.png` (1 px por célula, no espaço da
grade), `--no-elevation`. Com os padrões, a construção leva uns 4 s depois dos downloads e gera
256 chunks (~90 KiB no total, o zstd adora oceano).

### Fontes

| Chave (`fetch --only`) | Dado | URL | Licença |
|---|---|---|---|
| `ne_10m_admin_0_countries` | polígonos de países (máscara de terra, entidades `pais:XX`) | <https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip> | Natural Earth, domínio público |
| `ne_10m_admin_0_boundary_lines_land` | fronteiras terrestres (`limite_pais`) | <https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_boundary_lines_land.zip> | Natural Earth, domínio público |
| `ne_10m_rivers_lake_centerlines` | rios (`rio`, `scalerank ≤ 6`) | <https://naciscdn.org/naturalearth/10m/physical/ne_10m_rivers_lake_centerlines.zip> | Natural Earth, domínio público |
| `ne_10m_lakes` | lagos (`agua`, `scalerank ≤ 3` ou área ≥ 2 células) | <https://naciscdn.org/naturalearth/10m/physical/ne_10m_lakes.zip> | Natural Earth, domínio público |
| `ne_10m_roads` | rodovias (`rodovia`, tipos `Major Highway` e `Beltway`) | <https://naciscdn.org/naturalearth/10m/cultural/ne_10m_roads.zip> | Natural Earth, domínio público |
| `ibge_uf_2022` | malha de UFs 2022 (`estado:NN`, `limite_estado`) | <https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2022/Brasil/BR/BR_UF_2022.zip> | IBGE, dados abertos (uso livre com citação da fonte) |
| `ibge_biomas` | biomas 1:250.000 (cobertura do terreno) | <https://geoftp.ibge.gov.br/informacoes_ambientais/estudos_ambientais/biomas/vetores/Biomas_250mil.zip> | IBGE, dados abertos (uso livre com citação da fonte) |
| `etopo2022` (opcional) | relevo ETOPO 2022 exportado pelo ImageServer da NOAA, bbox −82..−30 / −36..13, 2400 px | <https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_all/ImageServer/exportImage?bbox=-82,-36,-30,13&bboxSR=4326&size=2400,2260&imageSR=4326&format=tiff&pixelType=F32&f=image> | NOAA ETOPO 2022, domínio público |

Para o Natural Earth há um espelho em `https://naturalearth.s3.amazonaws.com/10m_cultural/...`
(e `10m_physical`) que o `fetch` tenta quando o CDN falha. O cache fica em `data/raw/`
(gitignored, só o `.gitkeep` entra no git); arquivo presente é pulado sem checar hash, apague-o ou
use `--force` para baixar de novo. Cada download tem timeout (`--timeout`, padrão 300 s) e, quando
uma fonte obrigatória falha em todos os URLs, o comando aborta mostrando URL e erro. O relevo é
opcional: se não baixar, o nível sai sem `montanha` e com um aviso.

### Como a grade é amostrada

Tudo em `EPSG:5880` (SIRGAS 2000 / Brazil Polyconic) com os eixos girados 45°, como manda o
`chunk.md`: o `+x` da grade aponta para **sudeste** e o `+y` para **sudoeste** no terreno, então o
norte fica para cima na tela e a célula `(0, 0)` é a mais ao norte. Para a célula `(x, y)`,
`E = oe + (x − y)·c/√2` e `N = on − (x + y)·c/√2`, com `c = cell_meters` e `(oe, on)` o vértice
norte da célula `(0, 0)` (a "origem" do `[projection]`), escolhida de modo que o centro da célula
`(size/2, size/2)` caia no centro pedido (`--center-lon/--center-lat`). Isso é uma afim rotacionada
que o GDAL aceita direto: os vetores são rasterizados (`rasterio.features.rasterize`) e o relevo
reprojetado (`rasterio.warp.reproject`) **já na grade girada**, sem raster intermediário norte-acima.
O `--preview` é por isso uma imagem no espaço da grade: o Brasil aparece girado 45° (o norte é o
canto superior esquerdo).

Regras, todas vetorizadas em numpy (`level1.py`):

- **Terreno**: `oceano` onde não há polígono de país. Dentro do Brasil (malha do IBGE), o bioma
  decide: Amazônia e Mata Atlântica → `floresta`, Cerrado → `savana`, Caatinga → `caatinga`,
  Pampa → `campo`, Pantanal → `pantanal`. Terra fora do Brasil → `campo`. Relevo acima de 2000 m
  (em qualquer país) → `montanha`, o que pinta os Andes. Lagos → `agua`. Célula de terra com
  `oceano` em algum vizinho N/E/S/W vira `areia` (praia de uma célula).
- **Overlay**: `rio`, `rodovia`, `limite_estado` (calculado entre células de UFs diferentes, uma
  célula de largura, sem contornar a costa nem a fronteira) e `limite_pais` (linhas do Natural
  Earth). Quando se sobrepõem: `limite_pais` > `limite_estado` > `rodovia` > `rio`.
- **Objetos**: `arvore` em ~3% das células de `floresta`, escolhidas por hash da coordenada
  (determinístico).
- **Entidades** (camada `entity`): células do Brasil → `estado:<CD_UF>` (nome `NM_UF`); outras
  terras → `pais:<ISO alpha-2>` (`ISO_A2`, senão `ISO_A2_EH`, senão `ADM0_A3`; nome `NAME_PT`).
  O índice é atribuído na ordem da primeira aparição varrendo a grade linha a linha, e a lista sai
  em `[[entities]]` no `level.toml`. Oceano = 0.

### Limitações conhecidas

- Resolução de 7 km por célula: ilhas pequenas, rios estreitos e baías somem; a costa é a do
  Natural Earth 1:10M (fora do Brasil) e do IBGE (no Brasil), e as duas discordam em alguns km na
  fronteira. Faixas onde o contorno do Natural Earth passa do IBGE são absorvidas pela UF vizinha;
  só os Penedos de São Pedro e São Paulo sobram como `pais:BR` (não estão na malha de UFs).
- A Guiana Francesa é parte da França no Natural Earth, logo vira `pais:FR` ("França").
- Fora do Brasil não há classificação de cobertura: tudo é `campo` (ou `montanha` pelo relevo).
  Dentro do Brasil a cobertura é o bioma, não o uso do solo: `agricultura`, `pastagem` e `urbano`
  ainda não são emitidos (MapBiomas fica para uma fase seguinte).
- Rios, rodovias e fronteiras são rasterizados com `all_touched`, então diagonais saem em
  escadinha de até duas células; o tileset ainda não tem variantes de curva para overlay.
- Sem capitais/cidades (`predio`, `casa`): não há fonte limpa sem trabalho extra.
- O relevo vem de um export de ~2,4 km/px reamostrado por média; picos isolados abaixo de 2000 m
  na média da célula não viram `montanha` (o Brasil inteiro fica sem montanha, o que bate com a
  realidade a esta escala).

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
  sources.py  # fontes públicas do nível 1 e cache de download em data/raw/
  level1.py   # nível 1 real: grade rotacionada em EPSG:5880, rasterização, regras, chunks, preview
  cli.py      # typer: fetch, build-level1, chunk-test, inspect, autotile-demo, version
tests/        # pytest: round-trip, cabeçalho, máscaras, tileset, CLI, builder em GeoDataFrames sintéticos
```

Identificadores e comentários no código em inglês; documentação em PT-BR.

## Próximos passos

- **Cobertura por uso do solo (MapBiomas).** Trocar bioma por classificação real dentro do Brasil
  (`agricultura`, `pastagem`, `urbano`, água de reservatório) e dar cobertura aos vizinhos.
- **Níveis 2 e 3.** Mesma máquina (`level1.py` generaliza bem: muda a entidade, o centro, a célula e
  as fontes), com malhas municipais do IBGE para `limite_municipio` e entidades `municipio:NNNNNNN`.
- **Autotiling de overlay por máscara.** Rodovias, rios e ferrovias já passam pelo mesmo caminho
  de `[[autotile]]`, mas o placeholder ainda desenha uma faixa reta; curvas e ramificações entram
  quando o tileset tiver as variantes.
- **Hot paths em Rust.** Reamostragem e classificação em escala de município (1024² células × milhares
  de municípios) vão ficar lentas em Python puro; a ideia é mover esses trechos para uma extensão
  em Rust (PyO3) mantendo a orquestração em Python.

## Licença

Dupla licença MIT ou Apache-2.0, à sua escolha. Veja `LICENSE-MIT` e `LICENSE-APACHE`.
