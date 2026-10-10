from pathlib import Path

import pytest

from bramapa_pipeline import tiles
from bramapa_pipeline.tileset import TilesetError, builtin_tileset, load_tileset, parse_tileset

FIXTURE = Path(__file__).parent / "fixtures" / "tileset-autotile"


def _doc(**overrides):
    doc = {
        "spec_version": 0,
        "tileset": {"id": "t", "version": "1"},
        "tiles": [
            {"id": 1, "name": "campo", "class": "terrain", "cover": "campo"},
            {"id": 2, "name": "agua", "class": "terrain", "frames": 4, "fps": 4, "cover": "agua"},
            {"id": 100, "name": "agua_m0", "class": "terrain", "frames": 4, "fps": 4, "cover": "agua"},
            {"id": 40, "name": "arvore", "class": "object"},
        ],
        "autotile": [{"cover": "agua", "default": 2, "masks": {"0": 100}}],
    }
    doc.update(overrides)
    return doc


def test_load_fixture_indexes_tiles_covers_and_rules():
    ts = load_tileset(FIXTURE)  # directory form
    assert (ts.id, ts.version) == ("bramapa-test", "0.0.1-test")
    assert ts.source == FIXTURE / "tiles.toml"
    assert load_tileset(FIXTURE / "tiles.toml").id == ts.id  # file form
    assert set(ts.tiles) == {*tiles.ALL_NAMES} - {0} | set(range(100, 116)) | set(range(120, 136))
    assert ts.tiles[2].frames == 4 and ts.tiles[2].cover == "agua"
    assert ts.tiles[40].cover is None and ts.tiles[40].tile_class == "object"
    assert ts.by_name["casa"].id == 42
    assert {t.id for t in ts.tiles_for_cover("terrain", "agua")} == {2, *range(100, 116)}
    assert ts.tiles_for_cover("overlay", "agua") == ()
    assert set(ts.autotile) == {"agua", "oceano", "rio"}
    agua = ts.autotile["agua"]
    assert agua.default == 2 and agua.masks == {m: 100 + m for m in range(15)}
    assert agua.tile_for_mask(3) == 103 and agua.tile_for_mask(15) == 2
    assert agua.lut() == [*range(100, 115), 2]
    assert ts.tile_name(0) == "vazio" and ts.tile_name(121) == "oceano_m01" and ts.tile_name(999) == "?999"
    assert ts.ids_for_class("object") == {40, 41, 42}


def test_lookups_without_rule():
    ts = parse_tileset(_doc())
    assert ts.tile_for_cover("terrain", "campo") == 1
    assert ts.object_id("arvore") == 40
    with pytest.raises(TilesetError, match="no terrain tile for cover 'lava'"):
        ts.tile_for_cover("terrain", "lava")
    with pytest.raises(TilesetError, match="no object tile named 'campo'"):
        ts.object_id("campo")  # exists, but is terrain
    # Two tiles share a cover but no [[autotile]] rule picks between them.
    ts2 = parse_tileset(_doc(autotile=[]))
    with pytest.raises(TilesetError, match="2 tiles .* but no \\[\\[autotile\\]\\] rule"):
        ts2.tile_for_cover("terrain", "agua")


def test_builtin_matches_placeholder_table():
    ts = builtin_tileset()
    assert (ts.id, ts.version) == ("bramapa-base", "0.0.1")
    assert ts.autotile == {}
    assert {i: t.name for i, t in ts.tiles.items()} == {k: v for k, v in tiles.ALL_NAMES.items() if k != 0}
    assert ts.tile_for_cover("terrain", "oceano") == tiles.OCEANO
    assert ts.tile_for_cover("overlay", "rio") == tiles.RIO
    assert ts.object_id("casa") == tiles.CASA
    assert ts.tile_name(tiles.AGUA) == "agua"


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda d: d.update(spec_version=1), "unsupported spec_version 1"),
        (lambda d: d.pop("tileset"), "missing \\[tileset\\]"),
        (lambda d: d["tiles"].append({"id": 1, "name": "x", "class": "terrain", "cover": "x"}), "duplicate tile id 1"),
        (lambda d: d["tiles"].append({"id": 0, "name": "x", "class": "terrain", "cover": "x"}), "id must be"),
        (lambda d: d["tiles"].append({"id": 9, "name": "x", "class": "decal", "cover": "x"}), "class must be"),
        (lambda d: d["tiles"].append({"id": 9, "name": "x", "class": "terrain"}), "require a non-empty 'cover'"),
        (lambda d: d["tiles"].append({"id": 9, "name": "x", "class": "object", "cover": "x"}), "not allowed on object"),
        (lambda d: d["tiles"].append({"id": 9, "name": "x", "class": "terrain", "cover": "x", "frames": 2}), "fps"),
        (lambda d: d["tiles"].append({"id": 9, "name": "x", "class": "terrain", "cover": "x", "fps": 3}), "fps"),
        (lambda d: d["autotile"][0]["masks"].update({"16": 2}), "mask keys"),
        (lambda d: d["autotile"][0]["masks"].update({"07": 2}), "mask keys"),
        (lambda d: d["autotile"][0]["masks"].update({"1": 999}), "unknown tile id 999"),
        (lambda d: d["autotile"][0]["masks"].update({"1": 1}), "has cover 'campo', rule is for 'agua'"),
        (lambda d: d["autotile"].append({"cover": "agua", "default": 2}), "duplicate \\[\\[autotile\\]\\]"),
        (lambda d: d["tiles"].append({"id": 9, "name": "campo", "class": "terrain", "cover": "y"}), "duplicate tile name"),
    ],
)
def test_validation_errors(mutate, match):
    doc = _doc()
    mutate(doc)
    with pytest.raises(TilesetError, match=match):
        parse_tileset(doc)


def test_load_missing(tmp_path):
    with pytest.raises(TilesetError, match="tileset not found"):
        load_tileset(tmp_path)
