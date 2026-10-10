import tomllib

import numpy as np
from typer.testing import CliRunner

from bramapa_pipeline import tiles
from bramapa_pipeline.chunk import read_chunk
from bramapa_pipeline.cli import app
from bramapa_pipeline.synth import chunkify, generate
from bramapa_pipeline.level import LevelManifest

runner = CliRunner()


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

    # Everything decodes, uses only the placeholder ids, and the grid reassembles.
    terrain_ids, overlay_ids, object_ids = set(), set(), set()
    for p in files:
        ch = read_chunk(p)
        assert ch.size == 64
        terrain_ids |= set(np.unique(ch.terrain).tolist())
        overlay_ids |= set(np.unique(ch.overlay).tolist())
        object_ids |= {o.tile_id for o in ch.objects}
    assert terrain_ids <= {0, *tiles.TERRAIN_NAMES}
    assert overlay_ids <= {0, *tiles.OVERLAY_NAMES}
    assert object_ids <= set(tiles.OBJECT_NAMES)
    assert tiles.OCEANO in terrain_ids and tiles.CAMPO in terrain_ids
    assert tiles.RIO in overlay_ids and tiles.RODOVIA in overlay_ids
    assert tiles.ARVORE in object_ids


def test_chunkify_is_lossless_and_deterministic():
    m = LevelManifest(level=1, entity="pais:BR", name="t", width=150, height=90, cell_meters=1.0)
    lv = generate(150, 90, seed=3)
    assert np.array_equal(lv.terrain, generate(150, 90, seed=3).terrain)
    chunks = chunkify(lv, m)
    assert len(chunks) == m.chunks_x * m.chunks_y == 3 * 2
    full = np.zeros((2 * 64, 3 * 64), dtype=np.uint16)
    n_obj = 0
    for ch in chunks:
        full[ch.cy * 64 : (ch.cy + 1) * 64, ch.cx * 64 : (ch.cx + 1) * 64] = ch.terrain
        n_obj += len(ch.objects)
    assert np.array_equal(full[:90, :150], lv.terrain)
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
