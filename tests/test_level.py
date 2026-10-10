import tomllib
from datetime import date

import pytest

from bramapa_pipeline.level import LevelManifest, parse_period_date, read_level_toml, write_level_toml


def test_level_toml_matches_spec_example(tmp_path):
    m = LevelManifest(
        level=3,
        entity="municipio:4314902",
        name="Porto Alegre",
        width=1024,
        height=1024,
        cell_meters=30.0,
        crs="EPSG:5880",
        period_start="2023",
        period_end="2023",
    )
    p = write_level_toml(tmp_path / "level.toml", m)
    doc = tomllib.loads(p.read_text(encoding="utf-8"))
    assert doc == {
        "spec_version": 0,
        "level": {"level": 3, "entity": "municipio:4314902", "name": "Porto Alegre"},
        "grid": {"width": 1024, "height": 1024, "chunk_size": 64, "cell_meters": 30.0},
        "projection": {"crs": "EPSG:5880", "origin_x": 0.0, "origin_y": 0.0, "rotation_deg": 45.0},
        "tileset": {"id": "bramapa-base", "version": "0.0.1"},
        "period": {"start": "2023", "end": "2023"},
    }
    assert isinstance(doc["grid"]["cell_meters"], float)
    assert read_level_toml(p) == m


def test_no_period_omits_table(tmp_path):
    m = LevelManifest(level=1, entity="pais:BR", name="Teste sintético", width=256, height=256, cell_meters=8000)
    p = write_level_toml(tmp_path / "level.toml", m)
    doc = tomllib.loads(p.read_text(encoding="utf-8"))
    assert "period" not in doc
    assert doc["level"]["name"] == "Teste sintético"
    assert read_level_toml(p).period_start is None


def test_chunk_grid_rounds_up():
    m = LevelManifest(level=1, entity="pais:BR", name="x", width=100, height=64, cell_meters=1.0)
    assert (m.chunks_x, m.chunks_y) == (2, 1)
    assert m.chunk_coords() == [(0, 0), (1, 0)]


def test_parse_period_date():
    assert parse_period_date("2023", end=False) == date(2023, 1, 1)
    assert parse_period_date("2023", end=True) == date(2023, 12, 31)
    assert parse_period_date("2023-02", end=True) == date(2023, 2, 28)
    assert parse_period_date("2023-12", end=True) == date(2023, 12, 31)
    assert parse_period_date("2023-05-17", end=True) == date(2023, 5, 17)


def test_validation():
    with pytest.raises(ValueError):
        LevelManifest(level=4, entity="pais:BR", name="x", width=1, height=1, cell_meters=1.0)
    with pytest.raises(ValueError):
        LevelManifest(level=1, entity="BR", name="x", width=1, height=1, cell_meters=1.0)
    with pytest.raises(ValueError):
        LevelManifest(level=1, entity="pais:BR", name="x", width=1, height=1, cell_meters=1.0, period_start="2020")
