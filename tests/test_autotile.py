from pathlib import Path

import numpy as np
import pytest

from bramapa_pipeline.autotile import (
    MASK_E,
    MASK_N,
    MASK_S,
    MASK_W,
    apply_autotile,
    describe_masks,
    empty_cover_grid,
    encode_covers,
    neighbour_mask,
)
from bramapa_pipeline.tileset import TilesetError, builtin_tileset, load_tileset

FIXTURE = Path(__file__).parent / "fixtures" / "tileset-autotile"


def _reference_mask(grid: np.ndarray) -> np.ndarray:
    """Scalar transcription of the spec: N=(x,y-1) E=(x+1,y) S=(x,y+1) W=(x-1,y); outside = same."""
    h, w = grid.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            m = 0
            for bit, (dx, dy) in ((MASK_N, (0, -1)), (MASK_E, (1, 0)), (MASK_S, (0, 1)), (MASK_W, (-1, 0))):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < w and 0 <= ny < h) or grid[ny, nx] == grid[y, x]:
                    m |= bit
            out[y, x] = m
    return out


# Hand-built 5x5 with two covers: across its 25 cells every one of the 16 masks shows up.
GRID_5X5 = np.array(
    [
        list("abaab"),
        list("baaaa"),
        list("aaaba"),
        list("babab"),
        list("aaaba"),
    ]
)


def test_mask_bits_and_axes():
    g = np.full((3, 3), "b")
    g[1, 1] = "a"
    assert neighbour_mask(encode_covers(g)[0])[1, 1] == 0  # isolated
    g[0, 1] = "a"  # neighbour at (x, y-1) is N
    assert neighbour_mask(encode_covers(g)[0])[1, 1] == MASK_N
    g[0, 1], g[1, 2] = "b", "a"  # (x+1, y) is E
    assert neighbour_mask(encode_covers(g)[0])[1, 1] == MASK_E
    g[1, 2], g[2, 1] = "b", "a"  # (x, y+1) is S
    assert neighbour_mask(encode_covers(g)[0])[1, 1] == MASK_S
    g[2, 1], g[1, 0] = "b", "a"  # (x-1, y) is W
    assert neighbour_mask(encode_covers(g)[0])[1, 1] == MASK_W


def test_out_of_grid_counts_as_same_cover():
    g = np.full((3, 3), "b")
    g[0, 0] = "a"
    m = neighbour_mask(encode_covers(g)[0])
    assert m[0, 0] == MASK_N | MASK_W  # 9: both outside neighbours "match"
    assert m[0, 1] == MASK_N | MASK_E | MASK_S  # 7: N outside, E/S same, W is 'a'
    assert m[2, 2] == 15  # b surrounded by b and the border
    assert neighbour_mask(np.zeros((4, 7), dtype=int)).tolist() == np.full((4, 7), 15).tolist()


def test_5x5_reaches_all_16_masks_and_matches_reference():
    codes, names = encode_covers(GRID_5X5)
    assert names == ["", "a", "b"]
    m = neighbour_mask(codes)
    assert m.dtype == np.uint8
    assert np.array_equal(m, _reference_mask(GRID_5X5))
    assert set(m.ravel().tolist()) == set(range(16))
    # Spot checks against the spec bit assignment.
    assert m[0, 0] == MASK_N | MASK_W  # 'a' corner: E and S are 'b'; N/W outside count as same
    assert m[1, 0] == MASK_W  # 'b' on the west edge: only the outside neighbour "matches"
    assert m[1, 2] == 15  # 'a' surrounded by 'a'


def test_encode_covers_variants():
    # String grid with None/"" as empty.
    g = np.array([[None, "agua"], ["", "campo"]], dtype=object)
    codes, names = encode_covers(g)
    assert names[0] == "" and codes[0, 0] == 0 and codes[1, 0] == 0
    assert names[codes[0, 1]] == "agua" and names[codes[1, 1]] == "campo"
    # No empty cell at all: code 0 is still reserved for "".
    codes, names = encode_covers(np.array([["x", "y"]]))
    assert names == ["", "x", "y"] and codes.tolist() == [[1, 2]]
    # Integer grid + mapping; two codes may share a cover and then count as the same neighbour.
    codes, names = encode_covers(np.array([[0, 7, 9], [7, 7, 3]]), {7: "agua", 9: "agua", 3: "campo"})
    assert names == ["", "campo", "agua"]  # code order: 3 before 7/9
    assert codes.tolist() == [[0, 2, 2], [2, 2, 1]]
    with pytest.raises(ValueError, match="codes"):
        encode_covers(np.zeros((2, 2), dtype=int))
    with pytest.raises(ValueError, match="2-D"):
        encode_covers(np.array(["a", "b"]))


def test_rule_lookup_with_default_fallback():
    ts = load_tileset(FIXTURE)
    g = empty_cover_grid(3, 5)
    g[:] = "campo"
    g[1, 1:4] = "agua"  # a horizontal 3-cell lake, nothing touching the border
    ids = apply_autotile(g, ts, "terrain")
    assert ids.dtype == np.uint16
    assert ids[1].tolist() == [1, 100 + MASK_E, 100 + MASK_E + MASK_W, 100 + MASK_W, 1]
    # Fully surrounded water has mask 15, which the fixture leaves to `default` (base id 2).
    g[:] = "agua"
    assert set(np.unique(apply_autotile(g, ts, "terrain")).tolist()) == {2}
    # Same for the ocean rule, including the border.
    g[:] = "oceano"
    g[1, 2] = "campo"
    ids = apply_autotile(g, ts, "terrain")
    assert ids[1, 2] == 1 and ids[0, 0] == 3 and ids[0, 2] == 120 + 15 - MASK_S and ids[1, 1] == 120 + 15 - MASK_E


def test_covers_without_rule_and_errors():
    ts = load_tileset(FIXTURE)
    g = empty_cover_grid(2, 3)
    g[0] = "campo", "floresta", ""
    g[1] = "areia", "", "montanha"
    assert apply_autotile(g, ts).tolist() == [[1, 4, 0], [5, 0, 7]]
    assert not apply_autotile(empty_cover_grid(4, 4), ts).any()
    g[0, 0] = "lava"
    with pytest.raises(TilesetError, match="no terrain tile for cover 'lava'"):
        apply_autotile(g, ts)
    # A terrain cover on the overlay layer is also an error (class-aware lookup).
    with pytest.raises(TilesetError, match="no overlay tile for cover 'floresta'"):
        apply_autotile(g[:, 1:], ts, "overlay")


def test_overlay_layer_with_and_without_rules():
    ts = load_tileset(FIXTURE)
    g = empty_cover_grid(3, 4)
    g[0, :] = "rodovia"  # no rule -> single tile
    g[2, :] = "rio"  # rule present but every mask maps to 21
    ids = apply_autotile(g, ts, "overlay")
    assert ids.tolist() == [[20] * 4, [0] * 4, [21] * 4]
    # The builtin tileset has no rules at all: one id per cover on both layers.
    bt = builtin_tileset()
    assert apply_autotile(g, bt, "overlay").tolist() == ids.tolist()
    t = empty_cover_grid(2, 2)
    t[:] = "agua"
    t[0, 0] = "campo"
    assert apply_autotile(t, bt, "terrain").tolist() == [[1, 2], [2, 2]]


def test_int_coded_grid_matches_string_grid():
    ts = load_tileset(FIXTURE)
    names = {1: "campo", 2: "agua", 3: "oceano"}
    rng = np.random.default_rng(0)
    coded = rng.integers(0, 4, size=(12, 9))
    as_str = np.vectorize(lambda c: names.get(int(c), ""))(coded)
    assert np.array_equal(apply_autotile(coded, ts, codes=names), apply_autotile(as_str, ts))


def test_describe_masks():
    g = np.array([list("bbb"), list("bab"), list("bbb")])
    assert describe_masks(g, "a") == ["...", ".0.", "..."]
    assert describe_masks(g, "b") == ["fbf", "d.7", "fef"]  # corners 15; (0,1) misses S, (1,0) misses E
    with pytest.raises(TilesetError, match="not present"):
        describe_masks(g, "c")
