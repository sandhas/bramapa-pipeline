# bramapa-pipeline

Pipeline offline do **Bramapa**, um atlas 2.5D isométrico em pixel art do Brasil. Este repo é a
parte em Python que pega geodados (relevo, uso do solo, malhas do IBGE...) e cospe os arquivos que
o motor renderiza: chunks de tiles (`.bmc`) e o manifesto de cada nível (`level.toml`).

**Estado atual: Fase 0.** O esqueleto funciona de ponta a ponta (codec do `.bmc`, escrita de
`level.toml`, CLI, testes), mas o terreno ainda é sintético. Nada de geodado de verdade entra aqui
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
```

Escreve `level.toml` (nível 1, `pais:BR`, "Teste sintético", `cell_meters = 8000`, `EPSG:5880`,
tileset `bramapa-base@0.0.1`, sem período) e um `chunks/<cx>_<cy>.bmc` para cada chunk de 64×64.
O terreno vem de value noise em numpy: um continente "blobby" cercado de oceano, com areia na costa,
lagos, campo, floresta, savana e montanha; um trecho urbano pequeno; duas rodovias e um rio no
overlay; árvores na floresta e prédios/casas na cidade como objetos. Com a mesma semente o resultado
é idêntico byte a byte.

### Inspecionar

```sh
uv run bramapa-pipeline inspect ../bramapa-data-test                  # tabela com todos os chunks
uv run bramapa-pipeline inspect ../bramapa-data-test/chunks/1_1.bmc   # cabeçalho + stats de um chunk
```

## Ids de tile provisórios

Enquanto o `bramapa-tileset` não publica o `tiles.toml` real, os quatro repos usam esta convenção
(em `src/bramapa_pipeline/tiles.py`). `0` é sempre vazio.

| Camada | Ids |
|---|---|
| terrain | 1 campo, 2 agua, 3 oceano, 4 floresta, 5 areia, 6 urbano, 7 montanha, 8 savana, 9 agricultura, 10 pastagem, 11 caatinga, 12 pantanal |
| overlay | 20 rodovia, 21 rio, 22 limite_estado, 23 limite_municipio, 24 limite_pais, 25 ferrovia |
| objects | 40 arvore, 41 predio, 42 casa |

## Layout do código

```
src/bramapa_pipeline/
  chunk.py   # Chunk, encode/decode, write_chunk/read_chunk — o .bmc byte a byte
  level.py   # LevelManifest, write_level_toml/read_level_toml
  tiles.py   # ids provisórios acima
  synth.py   # gerador sintético (value noise) e chunkify()
  cli.py     # typer: chunk-test, inspect, version
tests/       # pytest: round-trip, offsets do cabeçalho, CLI
```

Identificadores e comentários no código em inglês; documentação em PT-BR.

## Próximos passos

- **Fase 2 — nível 1 (país) com dados reais.** Natural Earth para costa/relevo grosso, malhas do
  IBGE para limites de país/estado/município (`limite_*` no overlay), MapBiomas para a classificação
  de cobertura (floresta, savana, caatinga, campo, pantanal, agricultura, pastagem, urbano...).
  Entram `rasterio` e `geopandas` como dependências; a grade é amostrada em `EPSG:5880` com eixos
  rotacionados 45° como manda o `README.md` da spec.
- **Autotiling de verdade.** Hoje emitimos um id fixo por cobertura; quando o `tiles.toml` existir,
  o pipeline passa a resolver `(cover, máscara de vizinhança)` → tile pelas regras do tileset.
- **Hot paths em Rust.** Reamostragem e classificação em escala de município (1024² células × milhares
  de municípios) vão ficar lentas em Python puro; a ideia é mover esses trechos para uma extensão
  em Rust (PyO3) mantendo a orquestração em Python.

## Licença

Dupla licença MIT ou Apache-2.0, à sua escolha. Veja `LICENSE-MIT` e `LICENSE-APACHE`.
