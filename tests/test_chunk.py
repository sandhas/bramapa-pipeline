"""Codec tests against the byte layout in chunk.md (spec v0)."""

from __future__ import annotations

import struct
from datetime import date

import numpy as np
import pytest
import zstandard

from bramapa_pipeline.chunk import (
    HEADER_SIZE,
    NO_PERIOD,
    Chunk,
    ChunkFormatError,
    ChunkObject,
    date_to_days,
    days_to_date,
    decode,
    decode_header,
    encode,
    read_chunk,
    write_chunk,
)


def make_chunk(**kw) -> Chunk:
    rng = np.random.default_rng(0)
    terrain = rng.integers(0, 13, size=(64, 64), dtype=np.uint16)
    overlay = rng.integers(0, 26, size=(64, 64), dtype=np.uint16)
    return Chunk(cx=3, cy=-2, terrain=terrain, overlay=overlay, **kw)


def test_no_period_sentinel_is_i32_min():
    assert NO_PERIOD == -2147483648
    assert date_to_days(None) == NO_PERIOD
    assert days_to_date(NO_PERIOD) is None


def test_days_since_epoch():
    assert date_to_days(date(1970, 1, 1)) == 0
    assert date_to_days(date(2023, 1, 1)) == 19358
    assert days_to_date(19358) == date(2023, 1, 1)
    assert date_to_days(date(1969, 12, 31)) == -1


def test_header_layout_without_objects():
    ch = make_chunk()
    data = encode(ch)
    assert len(data) == HEADER_SIZE + struct.unpack_from("<I", data, 28)[0]

    assert data[0:4] == b"BMCK"
    assert struct.unpack_from("<H", data, 4)[0] == 0  # spec_version
    assert struct.unpack_from("<H", data, 6)[0] == 0  # flags: no objects
    assert struct.unpack_from("<H", data, 8)[0] == 64  # chunk_size
    assert data[10] == 2  # layer_count
    assert data[11] == 0  # reserved
    assert struct.unpack_from("<i", data, 12)[0] == 3  # cx
    assert struct.unpack_from("<i", data, 16)[0] == -2  # cy
    assert struct.unpack_from("<i", data, 20)[0] == NO_PERIOD
    assert struct.unpack_from("<i", data, 24)[0] == NO_PERIOD
    # Little-endian spelled out: i32::MIN is 00 00 00 80.
    assert data[20:24] == b"\x00\x00\x00\x80"


def test_header_layout_with_objects_and_period():
    ch = make_chunk(
        objects=[(1, 2, 40, 0), (63, 63, 41, 1)],
        period_start=date(2023, 1, 1),
        period_end=date(2023, 12, 31),
    )
    data = encode(ch)
    assert struct.unpack_from("<H", data, 6)[0] & 1 == 1
    assert data[10] == 3
    assert struct.unpack_from("<i", data, 20)[0] == 19358
    assert struct.unpack_from("<i", data, 24)[0] == 19722


def test_payload_layout():
    ch = make_chunk(objects=[(5, 7, 40, 1), (0, 0, 42, 0)])
    data = encode(ch)
    raw = zstandard.ZstdDecompressor().decompress(data[HEADER_SIZE:])
    n = 64 * 64
    assert len(raw) == 2 * n * 2 + 4 + 2 * 8
    terrain = np.frombuffer(raw[: n * 2], dtype="<u2").reshape(64, 64)
    overlay = np.frombuffer(raw[n * 2 : 2 * n * 2], dtype="<u2").reshape(64, 64)
    # Row-major: raw index y*64 + x.
    assert terrain[3, 10] == ch.terrain[3, 10]
    assert np.array_equal(terrain, ch.terrain)
    assert np.array_equal(overlay, ch.overlay)
    off = 2 * n * 2
    assert struct.unpack_from("<I", raw, off)[0] == 2
    assert struct.unpack_from("<BBHHH", raw, off + 4) == (5, 7, 40, 1, 0)
    assert struct.unpack_from("<BBHHH", raw, off + 12) == (0, 0, 42, 0, 0)


def test_roundtrip_plain():
    ch = make_chunk()
    back = decode(encode(ch))
    assert (back.cx, back.cy, back.size) == (3, -2, 64)
    assert np.array_equal(back.terrain, ch.terrain)
    assert np.array_equal(back.overlay, ch.overlay)
    assert back.objects == []
    assert back.period_start is None and back.period_end is None
    assert back.layer_count == 2


def test_roundtrip_objects_and_period():
    objs = [ChunkObject(1, 2, 40, 0), ChunkObject(63, 0, 41, 1), (10, 10, 42, 0)]
    ch = make_chunk(objects=objs, period_start=date(2010, 1, 1), period_end=date(2010, 12, 31))
    back = decode(encode(ch))
    assert back.objects == [ChunkObject(1, 2, 40, 0), ChunkObject(63, 0, 41, 1), ChunkObject(10, 10, 42, 0)]
    assert back.period_start == date(2010, 1, 1)
    assert back.period_end == date(2010, 12, 31)
    assert back.layer_count == 3
    # Encoding is deterministic, so re-encoding the decoded chunk is byte-identical.
    assert encode(back) == encode(ch)


def test_write_and_read_file(tmp_path):
    ch = make_chunk(objects=[(2, 2, 40, 0)])
    p = write_chunk(tmp_path / "chunks" / "3_-2.bmc", ch)
    assert p.exists()
    back = read_chunk(p)
    assert np.array_equal(back.terrain, ch.terrain)
    assert back.objects == [ChunkObject(2, 2, 40, 0)]


def test_rejects_bad_magic_and_version():
    data = bytearray(encode(make_chunk()))
    bad = bytes(b"XXXX") + bytes(data[4:])
    with pytest.raises(ChunkFormatError):
        decode_header(bad)
    data[4:6] = struct.pack("<H", 7)
    with pytest.raises(ChunkFormatError):
        decode_header(bytes(data))


def test_rejects_inconsistent_layer_count_and_truncation():
    data = bytearray(encode(make_chunk(objects=[(0, 0, 40, 0)])))
    data[10] = 2  # says 2 layers but flags bit 0 is set
    with pytest.raises(ChunkFormatError):
        decode(bytes(data))
    good = encode(make_chunk())
    with pytest.raises(ChunkFormatError):
        decode(good[:-3])
    with pytest.raises(ChunkFormatError):
        decode(good + b"\x00")


def test_validation():
    with pytest.raises(ChunkFormatError):
        Chunk(cx=0, cy=0, objects=[(64, 0, 40, 0)])
    with pytest.raises(ChunkFormatError):
        Chunk(cx=0, cy=0, terrain=np.zeros((32, 64), dtype=np.uint16))
    with pytest.raises(ChunkFormatError):
        Chunk(cx=0, cy=0, period_start=date(2020, 1, 1))


# ----------------------------------------------------------------------------- entity layer


def test_header_and_payload_with_entity_layer():
    entity = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) % 7
    ch = make_chunk(entity=entity)
    data = encode(ch)
    assert struct.unpack_from("<H", data, 6)[0] == 0x0002  # flag bit 1 only
    assert data[10] == 3  # terrain, overlay, entity
    raw = zstandard.ZstdDecompressor().decompress(data[HEADER_SIZE:])
    n = 64 * 64
    assert len(raw) == 3 * n * 2
    back_entity = np.frombuffer(raw[2 * n * 2 : 3 * n * 2], dtype="<u2").reshape(64, 64)
    assert np.array_equal(back_entity, entity)


def test_entity_layer_precedes_objects():
    entity = np.full((64, 64), 5, dtype=np.uint16)
    ch = make_chunk(entity=entity, objects=[(1, 1, 40, 0)])
    data = encode(ch)
    assert struct.unpack_from("<H", data, 6)[0] == 0x0003
    assert data[10] == 4
    raw = zstandard.ZstdDecompressor().decompress(data[HEADER_SIZE:])
    n = 64 * 64
    assert len(raw) == 3 * n * 2 + 4 + 8
    assert struct.unpack_from("<H", raw, 2 * n * 2)[0] == 5  # entity layer right after overlay
    assert struct.unpack_from("<I", raw, 3 * n * 2)[0] == 1  # then the objects count


def test_roundtrip_entity():
    entity = np.zeros((64, 64), dtype=np.uint16)
    entity[10:20, 30:40] = 3
    ch = make_chunk(entity=entity, objects=[(2, 3, 40, 1)])
    back = decode(encode(ch))
    assert back.entity is not None
    assert np.array_equal(back.entity, entity)
    assert back.objects == [ChunkObject(2, 3, 40, 1)]
    assert back.layer_count == 4
    assert encode(back) == encode(ch)


def test_canonical_encoding_without_entity():
    # No entity and no objects: flags 0, layer_count 2, byte-identical to a fresh encode.
    ch = make_chunk()
    assert ch.entity is None
    data = encode(ch)
    assert struct.unpack_from("<H", data, 6)[0] == 0
    assert data[10] == 2
    back = decode(data)
    assert back.entity is None
    assert encode(back) == data
    # An all-zero entity array is *present* (not canonical-equivalent to None).
    data2 = encode(make_chunk(entity=np.zeros((64, 64), dtype=np.uint16)))
    assert struct.unpack_from("<H", data2, 6)[0] == 0x0002
    assert decode(data2).entity is not None


def test_entity_layer_count_mismatch_and_shape():
    data = bytearray(encode(make_chunk(entity=np.zeros((64, 64), dtype=np.uint16))))
    data[10] = 2  # flags say entity present but layer_count says 2
    with pytest.raises(ChunkFormatError):
        decode(bytes(data))
    with pytest.raises(ChunkFormatError):
        Chunk(cx=0, cy=0, entity=np.zeros((32, 64), dtype=np.uint16))
