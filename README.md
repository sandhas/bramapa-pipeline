# bramapa-pipeline

Pipeline offline do **Bramapa**, um atlas 2.5D isométrico em pixel art do Brasil. Este repo é a
parte em Python que pega geodados (relevo, uso do solo, malhas do IBGE...) e cospe os arquivos que
o motor renderiza: chunks de tiles (`.bmc`) e o manifesto de cada nível (`level.toml`).

**Estado atual: Fase 4.** O esqueleto funciona de ponta a ponta (codec do `.bmc` com as camadas
`terrain`, `overlay`, `entity` e `objects`, escrita de `level.toml`, CLI, testes), os ids de tile
saem do `tiles.toml` de um tileset via autotiling, o **nível 1 (Brasil e vizinhos) é gerado de
geodados públicos** (Natural Earth, IBGE, ETOPO) por `build-level1` e os **níveis 2 (um estado) e 3
(um município) saem do MapBiomas 30 m, das malhas municipais do IBGE e do OpenStreetMap** por
`build-level2`/`build-level3`, todos no layout de diretórios da spec (`--root`). O gerador sintético
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
uv run bramapa-pipeline build-level1 --root ../bramapa-data --tileset ../bramapa-tileset   # -> ../bramapa-data/1/pais_BR/
uv run bramapa-pipeline build-level1 --out /tmp/l1 --tileset ../bramapa-tileset --preview /tmp/l1.png
uv run bramapa-pipeline inspect ../bramapa-data/1/pais_BR --tileset ../bramapa-tileset
```

Os três `build-level*` escrevem em `<root>/<nível>/<entidade>/` (`1/pais_BR`, `2/estado_43`,
`3/municipio_4314902`; o `:` do id vira `_` no disco, como manda o `chunk.md`). `--root` vale
`data/levels` (gitignored) por padrão e `--out DIR` sobrepõe tudo e grava direto em `DIR`.
`inspect <root>` lista todos os níveis encontrados na raiz.

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

## Níveis 2 e 3

```sh
uv run bramapa-pipeline build-level2 --uf 43 --tileset ../bramapa-tileset --preview /tmp/rs.png            # -> data/levels/2/estado_43/
uv run bramapa-pipeline build-level3 --municipio 4314902 --tileset ../bramapa-tileset --preview /tmp/poa.png  # -> data/levels/3/municipio_4314902/
uv run bramapa-pipeline inspect data/levels --tileset ../bramapa-tileset
```

Os dois precisam do `fetch` do nível 1 antes (Natural Earth e `BR_UF_2022`); o resto é baixado sob
demanda para o cache. A grade é a mesma do nível 1 (`EPSG:5880` girada 45°, origem no vértice norte
da célula `(0,0)`), só mudam centro, célula e fontes.

### `build-level2 --uf NN`

Um estado em ~1024×1024 células. O centro é o do losango envolvente do estado no espaço girado e a
célula, sem `--cell-meters`, é a menor que faz o estado inteiro caber em `--size` células
(`region.fit_cell_meters`: `r·√2·1,04 / size`, arredondada a 10 m, **mínimo 700 m**). O RS cai
exatamente no mínimo (700 m); um estado menor como Sergipe ficaria com células de 700 m em uma
grade sobrando, e um gigante como o Amazonas vai a ~2,3 km.

- **Terreno**: classes do MapBiomas agregadas pela tabela `mapbiomas.MAPBIOMAS_COVER`
  (3/49/9/6 → `floresta`, 4 → `savana`, 5/11/32 → `pantanal`, 12/50/25/30/13 → `campo`,
  15/21 → `pastagem`, lavouras 18–20/35/36/39–41/46–48/62 → `agricultura`, 23 → `areia`,
  24 → `urbano`, 29 → `montanha`, 33/31 → `agua`). A leitura é decimada pelos overviews do COG
  (~4 px de fonte por célula) e reamostrada por **moda** direto na grade girada. Nodata (fora do
  Brasil) vira `campo` em terra e `oceano` fora dela; água do MapBiomas fora da máscara de terra
  (a faixa costeira que o MapBiomas pinta como classe 33) vira `oceano`. Lagos do Natural Earth só
  entram onde o MapBiomas não cobre (lado uruguaio da Lagoa Mirim). Praia de uma célula como no nível 1.
- **Entidades**: `municipio:<CD_MUN>` (nome `NM_MUN`) para cada município da malha do IBGE da UF
  (todos os 497 do RS; municípios menores que uma célula recebem a célula do seu ponto
  representativo); as duas "lagoas-município" da malha (`4300001`, `4300002`) viram `estado:43`.
  Fora da UF: `estado:NN` vizinhos pela `BR_UF_2022` e `pais:XX` pelo Natural Earth. Oceano = 0.
- **Overlay**: `rio` (Natural Earth, `scalerank ≤ 8`), `ferrovia` e `rodovia` (OSM), `limite_municipio`
  (entre municípios diferentes, uma célula de largura), `limite_estado`, `limite_pais`; prioridade
  nessa ordem crescente.
- **Objetos**: `arvore` em ~3% da `floresta`, `predio` em ~6% e `casa` em ~12% do `urbano`, por hash
  da coordenada (determinístico).

### `build-level3 --municipio NNNNNNN`

Um município a 30 m (a resolução do MapBiomas). O lado da grade é o menor múltiplo de 64 que cabe o
losango do município (`region.fit_size`), até 2048; se nem 2048 células bastarem a 30 m, a célula
sobe e o comando avisa. Porto Alegre dá 1472×1472 (23×23 chunks). Opções: `--cell-meters 30`,
`--size` (força o lado), `--centre-lon/--centre-lat` (centro histórico para a densidade de prédios;
Porto Alegre tem o seu embutido em `level3.HISTORIC_CENTRES`, os outros usam o centróide das
células urbanas), `--no-osm`.

- **Terreno**: MapBiomas nativo (janela de ~2500×2200 px), mesma tabela; `oceano` só onde o
  MapBiomas é nodata **e** está fora da terra (malha municipal da UF + países do Natural Earth).
  O Guaíba e os lagos saem como `agua` (classe 33); os banhados do delta como `pantanal` (classe 11).
- **Entidades**: dentro do município = índice 1 (`municipio:NNNNNNN`), fora = 0, mas as células de
  fora continuam pintadas para o entorno existir. `limite_municipio` no contorno.
- **Overlay (OSM)**: `highway` motorway/trunk/primary/secondary → `rodovia`, `railway=rail` →
  `ferrovia`, `waterway=river` → `rio`. `tertiary`/`residential` entram também como `rodovia`
  quando a bbox tem menos de 60 mil dessas vias (`out count` antes de baixar; Porto Alegre tem
  ~32 mil, então entram); acima disso são puladas e o comando diz.
- **Objetos**: `predio` em células `urbano` com densidade caindo com a distância ao centro
  histórico (`region.predio_permille`: 25% até 2 km, linear até 8% a 5 km, até 2% a 10 km e além),
  `casa` em ~20% do `urbano` restante, `arvore` em ~10% da `floresta`.

### Fontes, licenças e cache

| Dado | Fonte | Licença / atribuição |
|---|---|---|
| Cobertura e uso da terra, 30 m | MapBiomas, GeoTIFF anual no bucket público `gs://mapbiomas-public/initiatives/brasil/` — coleção 9 (2023) para leituras decimadas (`.../collection_9/lclu/coverage/brasil_coverage_2023.tif`, única com overviews internos), coleção 11 (2025, `.../collection11/lulc/coverage/brazil_coverage/brazil_coverage-col11_2025.tif`) ou 10 (2024) para leituras a 30 m. Ordem de tentativa em `sources.MAPBIOMAS_COVERAGES` | **CC-BY-SA 4.0** — citar "Projeto MapBiomas – Coleção N da Série Anual de Mapas de Cobertura e Uso da Terra do Brasil"; o que deriva dela (os chunks) herda o *share-alike* |
| Malha municipal 2022 | IBGE, `geoftp.ibge.gov.br/.../municipio_2022/UFs/<UF>/<UF>_Municipios_2022.zip` (RS: 7,8 MB) | IBGE, dados abertos (citar a fonte) |
| Rodovias, ferrovias, rios | OpenStreetMap via Overpass API (`overpass-api.de`, espelhos `overpass.kumi.systems` e `overpass.private.coffee`), consulta OverpassQL com `out geom` | **ODbL 1.0** — todo produto que usa estes dados precisa trazer a atribuição "© OpenStreetMap contributors" (no motor/atlas, não só aqui), e uma base derivada redistribuída fica sob ODbL |
| Países, lagos, rios, fronteiras, rodovias de reserva | Natural Earth (já no cache do nível 1) | domínio público |

O MapBiomas **nunca é baixado inteiro** (cada arquivo tem 700–800 MB): o `mapbiomas.py` abre o COG
por `/vsicurl/` e lê só a janela da bbox do nível com *range requests*, usando os overviews quando
decimada (RS a 700 m: ~8,8 MB de rede, 40k×35k px lidos como 7,9k×6,9k; Porto Alegre a 30 m:
~0,9 MB). A janela vai para `data/raw/mapbiomas_<entidade>.tif` (RS 5,5 MB, POA 0,5 MB, com
coleção/ano nas tags) e reexecuções são offline. As respostas do Overpass ficam em
`data/raw/osm_<entidade>_*.json` (RS: 65 MB de rodovias+ferrovias, ~60 mil vias; Porto Alegre: 8,7 MB de
vias principais+ferrovia+rios e 21 MB de tertiary/residential). O ano do MapBiomas usado vai para
`[period]` do `level.toml` (e para o cabeçalho dos chunks).

Tempos medidos (cache frio, rede comum): RS ~70 s (quase tudo esperando o Overpass, que estava
sobrecarregado e precisou de tentativas), Porto Alegre ~55 s; com cache quente, 6 s e 8 s. A grade
em si leva 1–2 s.

### Limites

- Overpass é um serviço público e cai com frequência ("server too busy", HTTP 504/500). O `osm.py`
  roda nos três endpoints com até 3 passadas e espera crescente; se nada responder, o nível sai com
  as rodovias do Natural Earth (`Major Highway`, `Beltway`, `Secondary Highway`), sem ferrovia nem
  rio do OSM, e o comando avisa. Cache em disco evita bater de novo.
- Só a coleção 9 do MapBiomas tem overviews; as coleções 10 e 11 não, então o nível 2 fica em
  2023 e o nível 3 em 2025. Uma leitura decimada da coleção 10 funcionaria, mas puxa a janela
  inteira a 30 m (~2 min e centenas de MB para o RS).
- A tabela de classes é uma agregação grosseira: banhados e mangues viram `pantanal`, silvicultura
  vira `floresta`, mosaico de usos vira `pastagem`, afloramento rochoso vira `montanha`.
- Vias e contornos são rasterizados com `all_touched`: a 30 m uma rua de 10 m vira uma faixa de
  1–2 células; o tileset ainda não tem curvas de overlay.
- O nível 3 tem uma entidade só; os municípios vizinhos aparecem pintados, mas com entidade 0.
- Sem o `fetch` do nível 1 os comandos param com erro pedindo-o (precisam da `BR_UF_2022` e do
  Natural Earth).

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
  sources.py  # fontes públicas e cache de download em data/raw/ (NE, IBGE, MapBiomas, Overpass)
  level1.py   # nível 1 real: grade rotacionada em EPSG:5880, rasterização, regras, chunks, preview
  mapbiomas.py # tabela de classes MapBiomas -> cobertura, leitura remota por janela (/vsicurl/) e warp
  osm.py      # Overpass: consulta, cache JSON, parser de ways -> linhas, classificação de overlay
  region.py   # ajuste de grade (centro/célula/lado), objetos por distância, malhas do IBGE por UF
  level2.py   # nível 2 (estado): entidades municipais, MapBiomas decimado, overlays
  level3.py   # nível 3 (município): MapBiomas 30 m, OSM, prédios por distância ao centro
  cli.py      # typer: fetch, build-level1/2/3 (--root), chunk-test, inspect, autotile-demo, version
tests/        # pytest: round-trip, cabeçalho, máscaras, tileset, CLI, builders em dados sintéticos, Fase 4 (tabela, ajuste, layout, densidade, Overpass)
```

Identificadores e comentários no código em inglês; documentação em PT-BR.

## Próximos passos

- **MapBiomas no nível 1.** O nível 1 ainda pinta por bioma; a mesma leitura decimada do
  `mapbiomas.py` (a ~7 km, overview 256) daria `agricultura`, `pastagem` e `urbano` ao Brasil inteiro.
- **Todos os estados e municípios.** `build-level2`/`build-level3` fazem um por vez; um comando de
  lote (27 UFs, 5570 municípios) precisa de paralelismo e de respeito ao Overpass (ou de um extrato
  Geofabrik em vez da API).
- **Autotiling de overlay por máscara.** Rodovias, rios e ferrovias já passam pelo mesmo caminho
  de `[[autotile]]`, mas o placeholder ainda desenha uma faixa reta; curvas e ramificações entram
  quando o tileset tiver as variantes.
- **Hot paths em Rust.** Reamostragem e classificação em escala de município (1024² células × milhares
  de municípios) vão ficar lentas em Python puro; a ideia é mover esses trechos para uma extensão
  em Rust (PyO3) mantendo a orquestração em Python.

## Licença

Dupla licença MIT ou Apache-2.0, à sua escolha. Veja `LICENSE-MIT` e `LICENSE-APACHE`.
