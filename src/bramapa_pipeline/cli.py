"""``bramapa-pipeline`` command line."""

from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from . import SPEC_VERSION
from .autotile import apply_autotile, describe_masks, empty_cover_grid
from .chunk import HEADER_SIZE, Chunk, chunk_filename, days_to_date, read_chunk, read_header, write_chunk
from .level import LevelManifest, read_level_toml, write_level_toml
from .synth import chunkify, generate
from .tileset import Tileset, TilesetError, builtin_tileset, load_tileset

app = typer.Typer(
    help="Pipeline offline do Bramapa: gera e inspeciona chunks de tiles (.bmc).",
    no_args_is_help=True,
    add_completion=False,
)

TilesetOpt = Annotated[
    Path | None,
    typer.Option(
        "--tileset",
        help="Diretório do tileset (com tiles.toml). Sem ele, usa os ids provisórios embutidos.",
        exists=True,
        file_okay=True,
        dir_okay=True,
    ),
]


CacheDirOpt = Annotated[
    Path,
    typer.Option("--cache-dir", help="Diretório do cache de downloads (padrão data/raw, gitignored)."),
]


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_tileset(path: Path | None) -> Tileset:
    if path is None:
        return builtin_tileset()
    try:
        return load_tileset(path)
    except TilesetError as e:
        typer.echo(f"tileset inválido ({path}): {e}", err=True)
        raise typer.Exit(code=2) from e


@app.command("chunk-test")
def chunk_test(
    out: Annotated[Path, typer.Option("--out", help="Diretório do nível (recebe level.toml e chunks/).")],
    width: Annotated[int, typer.Option(help="Largura da grade em células.")] = 256,
    height: Annotated[int, typer.Option(help="Altura da grade em células.")] = 256,
    seed: Annotated[int, typer.Option(help="Semente do gerador sintético.")] = 42,
    tileset: TilesetOpt = None,
) -> None:
    """Escreve um nível sintético completo (level.toml + um .bmc por chunk) para testar o motor.

    Com --tileset, os ids vêm do tiles.toml indicado (autotiling incluído) e o level.toml aponta
    para esse tileset; sem ele, usa os ids provisórios embutidos (bramapa-base@0.0.1).
    """
    ts = _load_tileset(tileset)
    manifest = LevelManifest(
        level=1,
        entity="pais:BR",
        name="Teste sintético",
        width=width,
        height=height,
        cell_meters=8000.0,
        crs="EPSG:5880",
        tileset_id=ts.id,
        tileset_version=ts.version,
    )
    level = generate(width, height, seed=seed)
    chunks = chunkify(level, manifest, ts)

    out.mkdir(parents=True, exist_ok=True)
    write_level_toml(out / "level.toml", manifest)
    chunk_dir = out / "chunks"
    total_bytes = 0
    for ch in chunks:
        p = write_chunk(chunk_dir / chunk_filename(ch.cx, ch.cy), ch)
        total_bytes += p.stat().st_size

    n_obj = len(level.objects)
    typer.echo(
        f"{out}: {width}x{height} células, {manifest.chunks_x}x{manifest.chunks_y} chunks "
        f"({len(chunks)} arquivos, {total_bytes / 1024:.1f} KiB), {n_obj} objetos, seed={seed}, "
        f"tileset={ts.id}@{ts.version}{' (embutido)' if tileset is None else ''}"
    )


@app.command("fetch")
def fetch_cmd(
    cache_dir: CacheDirOpt = Path("data/raw"),
    only: Annotated[list[str] | None, typer.Option("--only", help="Baixar só estas fontes (chave).")] = None,
    force: Annotated[bool, typer.Option("--force", help="Baixar de novo mesmo se já estiver no cache.")] = False,
    timeout: Annotated[float, typer.Option(help="Timeout por download, em segundos.")] = 300.0,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Baixa os geodados públicos do nível 1 (Natural Earth, IBGE, ETOPO) para o cache em disco.

    Arquivo já presente no cache é pulado (sem checagem de hash). Fontes opcionais (relevo) só
    geram aviso quando falham; as obrigatórias abortam com o URL e o erro.
    """
    from .sources import SOURCES, SourceError, fetch_all

    _setup_logging(verbose)
    keys = tuple(only) if only else None
    if keys:
        bad = [k for k in keys if k not in SOURCES]
        if bad:
            typer.echo(f"fontes desconhecidas: {bad}; válidas: {list(SOURCES)}", err=True)
            raise typer.Exit(code=2)
    try:
        result = fetch_all(cache_dir, keys=keys, timeout=timeout, force=force)
    except SourceError as e:
        typer.echo(f"erro: {e}", err=True)
        raise typer.Exit(code=1) from e
    typer.echo("")
    typer.echo(f"  {'fonte':<36} {'bytes':>10}  arquivo")
    for key, path in result.items():
        src = SOURCES[key]
        if path is None:
            typer.echo(f"  {key:<36} {'-':>10}  (opcional, indisponível)  {src.licence}")
        else:
            typer.echo(f"  {key:<36} {path.stat().st_size:>10}  {path}  [{src.licence}]")


@app.command("build-level1")
def build_level1(
    out: Annotated[Path, typer.Option("--out", help="Diretório do nível (recebe level.toml e chunks/).")],
    tileset: TilesetOpt = None,
    cell_meters: Annotated[float, typer.Option(help="Tamanho da célula no terreno, em metros.")] = 7000.0,
    size: Annotated[int, typer.Option(help="Lado da grade em células.")] = 1024,
    center_lon: Annotated[float, typer.Option(help="Longitude do centro da grade.")] = -53.0,
    center_lat: Annotated[float, typer.Option(help="Latitude do centro da grade.")] = -14.0,
    cache_dir: CacheDirOpt = Path("data/raw"),
    preview: Annotated[Path | None, typer.Option("--preview", help="Escreve um PNG (1 px por célula) das coberturas.")] = None,
    no_elevation: Annotated[bool, typer.Option("--no-elevation", help="Ignora o relevo (sem montanha).")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Constrói o nível 1 real (Brasil e vizinhos) a partir dos geodados do cache.

    Precisa de `fetch` antes. Grade em EPSG:5880 com eixos rotacionados 45°: a célula (0,0) é a
    mais ao norte, +x aponta para sudeste e +y para sudoeste no terreno.
    """
    from .level1 import GridSpec, build_grid, load_inputs, write_level, write_preview

    _setup_logging(verbose)
    ts = _load_tileset(tileset)
    spec = GridSpec(size=size, cell_meters=cell_meters, center_lon=center_lon, center_lat=center_lat)
    t0 = time.perf_counter()
    try:
        inputs = load_inputs(spec, cache_dir, use_elevation=not no_elevation)
    except FileNotFoundError as e:
        typer.echo(f"erro: {e}", err=True)
        raise typer.Exit(code=1) from e
    t_load = time.perf_counter()
    grid = build_grid(spec, inputs)
    t_grid = time.perf_counter()
    manifest, n_chunks, total_bytes = write_level(grid, spec, ts, out)
    t_write = time.perf_counter()
    if preview is not None:
        write_preview(grid, preview)

    covers = Counter(grid.cover_names().ravel().tolist())
    cover_txt = ", ".join(f"{k}={v}" for k, v in covers.most_common())
    typer.echo(
        f"{out}: {size}x{size} células de {cell_meters:g} m, {manifest.chunks_x}x{manifest.chunks_y} chunks "
        f"({n_chunks} arquivos, {total_bytes / 1024:.1f} KiB), {len(grid.entities)} entidades, "
        f"{len(grid.objects)} objetos, tileset={ts.id}@{ts.version}"
    )
    typer.echo(f"  origem (E, N) = {manifest.origin_x:.1f}, {manifest.origin_y:.1f} em {manifest.crs}, rotação {manifest.rotation_deg:g}°")
    typer.echo(f"  coberturas: {cover_txt}")
    typer.echo(
        f"  tempos: leitura {t_load - t0:.1f}s, grade {t_grid - t_load:.1f}s, autotile+escrita {t_write - t_grid:.1f}s"
    )
    if preview is not None:
        typer.echo(f"  preview: {preview} (espaço da grade: o Brasil aparece girado 45°)")


def _layer_stats(arr: np.ndarray, ts: Tileset) -> str:
    nonzero = arr[arr != 0]
    ids = sorted(np.unique(nonzero).tolist())
    label = ", ".join(f"{i}={ts.tile_name(i)}" for i in ids) or "-"
    return f"{int(nonzero.size)} células não vazias; ids: {label}"


def _fmt_period(days: int) -> str:
    d = days_to_date(days)
    return f"{days} ({d.isoformat()})" if d is not None else f"{days} (sem período)"


def inspect_file(path: Path, ts: Tileset) -> None:
    h = read_header(path)
    size = path.stat().st_size
    typer.echo(f"{path}  ({size} bytes = {HEADER_SIZE} cabeçalho + {h.payload_len} carga zstd)")
    typer.echo(f"  spec_version  = {h.spec_version}")
    typer.echo(
        f"  flags         = {h.flags:#06x}  (objects={'sim' if h.has_objects else 'não'}, "
        f"entity={'sim' if h.has_entity else 'não'})"
    )
    typer.echo(f"  chunk_size    = {h.chunk_size}")
    typer.echo(f"  layer_count   = {h.layer_count}")
    typer.echo(f"  reserved      = {h.reserved}")
    typer.echo(f"  cx, cy        = {h.cx}, {h.cy}")
    typer.echo(f"  period_start  = {_fmt_period(h.period_start)}")
    typer.echo(f"  period_end    = {_fmt_period(h.period_end)}")
    typer.echo(f"  payload_len   = {h.payload_len}")
    ch: Chunk = read_chunk(path)
    typer.echo(f"  terrain : {_layer_stats(ch.terrain, ts)}")
    typer.echo(f"  overlay : {_layer_stats(ch.overlay, ts)}")
    if ch.entity is not None:
        nz = ch.entity[ch.entity != 0]
        idx = sorted(np.unique(nz).tolist())
        typer.echo(f"  entity  : {int(nz.size)} células com entidade; índices: {', '.join(map(str, idx)) or '-'}")
    else:
        typer.echo("  entity  : (camada ausente)")
    if ch.has_objects:
        counts = Counter(o.tile_id for o in ch.objects)
        label = ", ".join(f"{i}={ts.tile_name(i)}×{n}" for i, n in sorted(counts.items()))
        typer.echo(f"  objects : {len(ch.objects)} ({label})")
    else:
        typer.echo("  objects : (camada ausente)")


def inspect_level(level_dir: Path, ts: Tileset) -> None:
    manifest_path = level_dir / "level.toml"
    if manifest_path.exists():
        m = read_level_toml(manifest_path)
        period = f"período={m.period_start}..{m.period_end}" if m.period_start else "sem período"
        typer.echo(
            f'{manifest_path}: nível {m.level} {m.entity} "{m.name}" — {m.width}x{m.height} células, '
            f"chunk_size={m.chunk_size}, cell_meters={m.cell_meters}, crs={m.crs}, "
            f"tileset={m.tileset_id}@{m.tileset_version}, {period}"
        )
        if m.entities:
            typer.echo(f"  {len(m.entities)} entidades em [[entities]]: " + ", ".join(f"{i}={eid}" for i, eid, _ in m.entities[:12]) + (" ..." if len(m.entities) > 12 else ""))
        expected = set(m.chunk_coords())
    else:
        typer.echo(f"{level_dir}: sem level.toml, listando só os chunks")
        expected = set()

    chunk_dir = level_dir / "chunks" if (level_dir / "chunks").is_dir() else level_dir
    files = sorted(chunk_dir.glob("*.bmc"), key=lambda p: tuple(int(t) for t in p.stem.split("_")))
    if not files:
        typer.echo(f"  nenhum .bmc em {chunk_dir}")
        raise typer.Exit(code=1)

    typer.echo("")
    typer.echo(f"  {'cx':>4} {'cy':>4} {'bytes':>7} {'terrain':>8} {'overlay':>8} {'entity':>8} {'objects':>8}  ids")
    tot_terrain = tot_overlay = tot_objects = tot_bytes = tot_entity = 0
    entity_idx: set[int] = set()
    all_ids: Counter[int] = Counter()
    seen = set()
    for p in files:
        ch = read_chunk(p)
        seen.add((ch.cx, ch.cy))
        t_n = int(np.count_nonzero(ch.terrain))
        o_n = int(np.count_nonzero(ch.overlay))
        ids = set(np.unique(ch.terrain).tolist()) | set(np.unique(ch.overlay).tolist())
        ids |= {o.tile_id for o in ch.objects}
        ids.discard(0)
        all_ids.update(ids)
        size = p.stat().st_size
        if ch.entity is not None:
            e_n = int(np.count_nonzero(ch.entity))
            entity_idx |= set(np.unique(ch.entity).tolist()) - {0}
            e_txt = str(e_n)
        else:
            e_n, e_txt = 0, "-"
        tot_terrain += t_n
        tot_overlay += o_n
        tot_entity += e_n
        tot_objects += len(ch.objects)
        tot_bytes += size
        typer.echo(
            f"  {ch.cx:>4} {ch.cy:>4} {size:>7} {t_n:>8} {o_n:>8} {e_txt:>8} {len(ch.objects):>8}  "
            + ",".join(str(i) for i in sorted(ids))
        )
    typer.echo(
        f"  {'tot':>4} {len(files):>4} {tot_bytes:>7} {tot_terrain:>8} {tot_overlay:>8} {tot_entity:>8} {tot_objects:>8}"
    )
    if entity_idx:
        typer.echo(f"  índices de entidade usados: {len(entity_idx)} ({min(entity_idx)}..{max(entity_idx)})")
    typer.echo("")
    typer.echo("  tiles usados: " + ", ".join(f"{i}={ts.tile_name(i)}" for i in sorted(all_ids)))
    if expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        if missing:
            typer.echo(f"  chunks ausentes (tratados como vazios pelo motor): {missing}")
        if extra:
            typer.echo(f"  chunks fora da grade (inválidos): {extra}")
        if not missing and not extra:
            typer.echo(f"  grade completa: {len(expected)} chunks esperados, {len(seen)} presentes")


@app.command("inspect")
def inspect(
    path: Annotated[Path, typer.Argument(help="Um arquivo .bmc ou o diretório de um nível.")],
    tileset: TilesetOpt = None,
) -> None:
    """Mostra cabeçalho e estatísticas de um chunk, ou um resumo de todos os chunks de um nível.

    Com --tileset, os ids são nomeados pelo tiles.toml indicado (variantes de autotiling incluídas).
    """
    if not path.exists():
        typer.echo(f"não existe: {path}", err=True)
        raise typer.Exit(code=2)
    ts = _load_tileset(tileset)
    if path.is_dir():
        inspect_level(path, ts)
    else:
        inspect_file(path, ts)


@app.command("autotile-demo")
def autotile_demo(
    tileset: TilesetOpt = None,
    cover: Annotated[str, typer.Option(help="Cobertura do lago.")] = "agua",
    around: Annotated[str, typer.Option(help="Cobertura ao redor.")] = "campo",
) -> None:
    """Desenha um lago pequeno e mostra a máscara de vizinhança e o id escolhido por célula.

    Máscara: N=1 E=2 S=4 W=8, bit ligado = vizinho com a mesma cobertura; fora da grade = igual.
    """
    ts = _load_tileset(tileset)
    grid = empty_cover_grid(7, 11)
    grid[:] = around
    grid[1:6, 2:9] = cover  # a 7x5 lake...
    grid[1, 2] = grid[1, 8] = grid[5, 2] = grid[5, 8] = around  # ...with rounded corners
    grid[3, 4:6] = around  # and an island
    grid[2, 9] = grid[3, 9] = cover  # a bay poking east
    grid[0, 5] = cover  # and a cell touching the border (north counts as "same")
    try:
        ids = apply_autotile(grid, ts, "terrain")
    except TilesetError as e:
        typer.echo(f"erro: {e}", err=True)
        raise typer.Exit(code=1) from e
    rule = ts.autotile.get(cover)
    typer.echo(
        f"tileset {ts.id}@{ts.version}; cover {cover!r}: "
        + (
            f"default={rule.default}, {len(rule.masks)} máscaras explícitas"
            if rule
            else "sem regra [[autotile]], id único"
        )
    )
    typer.echo("")
    typer.echo("máscara (hex, N=1 E=2 S=4 W=8)   id do tile")
    sel = grid == cover
    for mrow, idrow, srow in zip(describe_masks(grid, cover), ids, sel):
        idtxt = " ".join(f"{int(i):>3}" if c else "  ." for i, c in zip(idrow, srow))
        typer.echo(f"  {mrow}   {idtxt}")
    used = sorted(set(int(i) for i in ids[sel].tolist()))
    typer.echo("")
    typer.echo("ids usados pelo lago: " + ", ".join(f"{i}={ts.tile_name(i)}" for i in used))


@app.command("version")
def version() -> None:
    """Mostra a versão do pacote e a spec_version implementada."""
    from importlib.metadata import version as pkg_version

    typer.echo(f"bramapa-pipeline {pkg_version('bramapa-pipeline')} (spec_version = {SPEC_VERSION})")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
