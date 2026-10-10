import tomllib
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from bramapa_pipeline import tiles
from bramapa_pipeline.chunk import read_chunk
from bramapa_pipeline.cli import app
from bramapa_pipeline.synth import chunkify, generate
from bramapa_pipeline.level import LevelManifest
from bramapa_pipeline.tileset import builtin_tileset, load_tileset

runner = CliRunner()
FIXTURE = Path(__file__).parent / "fixtures" / "tileset-autotile"


def _collect_ids(files):
    terrain_ids, overlay_ids, object_ids = set(), set(), set()
    for p in files:
        ch = read_chunk(p)
        assert ch.size == 64
        terrain_ids |= set(np.unique(ch.terrain).tolist())
        overlay_ids |= set(np.unique(ch.overlay).tolist())
        object_ids |= {o.tile_id for o in ch.objects}
    return terrain_ids, overlay_ids, object_ids


def test_chunk_test_writes_level(tmp_path):
    out = tmp_path / "lvl"
    r = runner.invoke(app, ["chunk-test", "--out", str(out), "--width", "200", "--height", "130", "--seed", "7"])
    assert r.exit_code == 0, r.output
    doc = tomllib.loads((out / "level.toml").read_text(encoding="utf-8"))
    assert doc["spec_version"] == 0
    assert doc["level"] == {"level": 1, "entity": "pais:BR", "name": "Teste sintético"}
    assert doc["grid"] == {"width": 200, "height": 130, "chunk_size": 64, "cell_meters": 8000.0}
    assert doc["projection"]["crs"] == "EPSG:5880"
    assert doc["tileset"] == {"id": "bramapa-base", "version": "0.0.1"}
    assert "period" not in doc

    files = sorted((out / "chunks").glob("*.bmc"))
    assert len(files) == 4 * 3  # ceil(200/64) x ceil(130/64)
    names = {p.name for p in files}
    assert "0_0.bmc" in names and "3_2.bmc" in names

    # Everything decodes and uses only the placeholder ids (builtin tileset has no variants).
    terrain_ids, overlay_ids, object_ids = _collect_ids(files)
    assert terrain_ids <= {0, *tiles.TERRAIN_NAMES}
    assert overlay_ids <= {0, *tiles.OVERLAY_NAMES}
    assert object_ids <= set(tiles.OBJECT_NAMES)
    assert tiles.OCEANO in terrain_ids and tiles.CAMPO in terrain_ids
    assert tiles.RIO in overlay_ids and tiles.RODOVIA in overlay_ids
    assert tiles.ARVORE in object_ids
    assert "tileset=bramapa-base@0.0.1 (embutido)" in r.output


def test_chunk_test_with_tileset_autotiles_shores(tmp_path):
    out = tmp_path / "lvl"
    args = ["chunk-test", "--out", str(out), "--width", "200", "--height", "130", "--seed", "7", "--tileset", str(FIXTURE)]
    r = runner.invoke(app, args)
    assert r.exit_code == 0, r.output
    assert "tileset=bramapa-test@0.0.1-test" in r.output and "embutido" not in r.output
    doc = tomllib.loads((out / "level.toml").read_text(encoding="utf-8"))
    assert doc["tileset"] == {"id": "bramapa-test", "version": "0.0.1-test"}

    ts = load_tileset(FIXTURE)
    terrain_ids, overlay_ids, object_ids = _collect_ids(sorted((out / "chunks").glob("*.bmc")))
    assert terrain_ids <= {0, *ts.ids_for_class("terrain")}  # only known ids
    assert overlay_ids <= {0, *ts.ids_for_class("overlay")}
    assert object_ids <= ts.ids_for_class("object")
    agua_variants = terrain_ids & set(range(100, 116))
    oceano_variants = terrain_ids & set(range(120, 136))
    assert agua_variants and oceano_variants  # shores got mask variants...
    assert 115 not in terrain_ids and 135 not in terrain_ids  # ...mask 15 is the base id
    assert tiles.AGUA in terrain_ids and tiles.OCEANO in terrain_ids  # ...and inland cells keep 2/3
    assert overlay_ids == {0, tiles.RIO, tiles.RODOVIA}  # rio rule maps everything to 21

    # Same seed, same covers: the terrain differs from the builtin run only where variants apply.
    lv = generate(200, 130, seed=7)
    m = LevelManifest(level=1, entity="pais:BR", name="t", width=200, height=130, cell_meters=1.0)
    base = {(c.cx, c.cy): c for c in chunkify(lv, m, builtin_tileset())}
    for c in chunkify(lv, m, ts):
        b = base[(c.cx, c.cy)].terrain
        diff = c.terrain != b
        assert np.isin(b[diff], [tiles.AGUA, tiles.OCEANO]).all()
        assert np.array_equal(c.overlay, base[(c.cx, c.cy)].overlay)

    # inspect names the variants when given the tileset, and shows "?" without it.
    r = runner.invoke(app, ["inspect", str(out), "--tileset", str(FIXTURE)])
    assert r.exit_code == 0, r.output
    assert "_m" in r.output and "bramapa-test@0.0.1-test" in r.output
    r = runner.invoke(app, ["inspect", str(out)])
    assert r.exit_code == 0 and "=?1" in r.output


def test_chunk_test_rejects_bad_tileset(tmp_path):
    bad = tmp_path / "ts"
    bad.mkdir()
    (bad / "tiles.toml").write_text("spec_version = 1\n[tileset]\nid='x'\nversion='1'\n", encoding="utf-8")
    r = runner.invoke(app, ["chunk-test", "--out", str(tmp_path / "lvl"), "--tileset", str(bad)])
    assert r.exit_code == 2
    assert "unsupported spec_version" in r.output
    assert not (tmp_path / "lvl").exists()


def test_autotile_demo():
    r = runner.invoke(app, ["autotile-demo"])
    assert r.exit_code == 0, r.output
    assert "sem regra [[autotile]]" in r.output and "ids usados pelo lago: 2=agua" in r.output
    r = runner.invoke(app, ["autotile-demo", "--tileset", str(FIXTURE)])
    assert r.exit_code == 0, r.output
    assert "default=2, 15 máscaras explícitas" in r.output
    assert ".....5....." in r.output  # the border cell: N outside + S water
    assert "105=agua_m05" in r.output and "2=agua" in r.output
    r = runner.invoke(app, ["autotile-demo", "--cover", "lava"])
    assert r.exit_code == 1


def test_chunkify_resolves_covers_and_names_through_tileset():
    lv = generate(150, 90, seed=3)
    assert lv.terrain.dtype.kind == "U" and lv.overlay.dtype.kind == "U"
    assert {"oceano", "campo"} <= set(np.unique(lv.terrain).tolist())
    assert set(np.unique(lv.overlay).tolist()) <= {"", "rio", "rodovia"}
    assert {o[2] for o in lv.objects} <= {"arvore", "predio", "casa"}
    m = LevelManifest(level=1, entity="pais:BR", name="t", width=150, height=90, cell_meters=1.0)
    chunks = chunkify(lv, m)  # builtin tileset by default
    ids = {o.tile_id for c in chunks for o in c.objects}
    assert ids <= {tiles.ARVORE, tiles.PREDIO, tiles.CASA}


def test_chunkify_is_lossless_and_deterministic():
    m = LevelManifest(level=1, entity="pais:BR", name="t", width=150, height=90, cell_meters=1.0)
    lv = generate(150, 90, seed=3)
    assert np.array_equal(lv.terrain, generate(150, 90, seed=3).terrain)
    chunks = chunkify(lv, m)
    expected = np.vectorize(lambda c: tiles.ALL_NAMES_INV.get(c, 0))(lv.terrain).astype(np.uint16)
    assert len(chunks) == m.chunks_x * m.chunks_y == 3 * 2
    full = np.zeros((2 * 64, 3 * 64), dtype=np.uint16)
    n_obj = 0
    for ch in chunks:
        full[ch.cy * 64 : (ch.cy + 1) * 64, ch.cx * 64 : (ch.cx + 1) * 64] = ch.terrain
        n_obj += len(ch.objects)
    assert np.array_equal(full[:90, :150], expected)
    assert not full[90:, :].any() and not full[:, 150:].any()  # padding is empty
    assert n_obj == len(lv.objects)


def test_inspect_file_and_dir(tmp_path):
    out = tmp_path / "lvl"
    assert runner.invoke(app, ["chunk-test", "--out", str(out), "--width", "128", "--height", "64"]).exit_code == 0
    r = runner.invoke(app, ["inspect", str(out / "chunks" / "1_0.bmc")])
    assert r.exit_code == 0, r.output
    assert "spec_version  = 0" in r.output
    assert "cx, cy        = 1, 0" in r.output
    assert "sem período" in r.output
    r = runner.invoke(app, ["inspect", str(out)])
    assert r.exit_code == 0, r.output
    assert "grade completa: 2 chunks esperados, 2 presentes" in r.output
    assert "3=oceano" in r.output


def test_inspect_missing_path(tmp_path):
    r = runner.invoke(app, ["inspect", str(tmp_path / "nope")])
    assert r.exit_code == 2
