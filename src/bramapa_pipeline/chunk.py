"""Codec for the ``.bmc`` chunk file (spec v0, ``chunk.md``).

Layout::

    32-byte little-endian header (uncompressed)
    zstd frame (standard, no dictionary) containing:
        terrain  : chunk_size² × u16, row-major (y outer, x inner)
        overlay  : chunk_size² × u16, same order
        objects  : only if header.flags & 1 — u32 count, then count × 8-byte records
                   (u8 cell_x, u8 cell_y, u16 tile_id, u16 flags, u16 reserved)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple

import numpy as np
import zstandard

from . import SPEC_VERSION

MAGIC = b"BMCK"
HEADER_SIZE = 32
DEFAULT_CHUNK_SIZE = 64

# "<" = little-endian, no padding. Fields follow the offset table in chunk.md.
HEADER_STRUCT = struct.Struct("<4sHHHBBiiiiI")
assert HEADER_STRUCT.size == HEADER_SIZE

FLAG_HAS_OBJECTS = 0x0001
"""Header flag bit 0: payload has an ``objects`` layer."""

OBJECT_FLAG_FLIP_H = 0x0001
"""Object flag bit 0: sprite mirrored horizontally."""

NO_PERIOD = -(2**31)
"""``i32::MIN``: sentinel for "no period" in ``period_start``/``period_end``."""

EPOCH = date(1970, 1, 1)

OBJECT_STRUCT = struct.Struct("<BBHHH")
assert OBJECT_STRUCT.size == 8
OBJECT_DTYPE = np.dtype(
    [
        ("cell_x", "<u1"),
        ("cell_y", "<u1"),
        ("tile_id", "<u2"),
        ("flags", "<u2"),
        ("reserved", "<u2"),
    ]
)


class ChunkFormatError(ValueError):
    """Raised when bytes do not form a valid ``.bmc`` chunk."""


class ChunkObject(NamedTuple):
    """One entry of the ``objects`` layer. Plain tuples ``(x, y, tile_id, flags)`` also work."""

    cell_x: int
    cell_y: int
    tile_id: int
    flags: int = 0


def date_to_days(d: date | None) -> int:
    """ISO date -> days since 1970-01-01; ``None`` -> :data:`NO_PERIOD`."""
    if d is None:
        return NO_PERIOD
    return (d - EPOCH).days


def days_to_date(days: int) -> date | None:
    """Days since 1970-01-01 -> date; :data:`NO_PERIOD` -> ``None``."""
    if days == NO_PERIOD:
        return None
    return EPOCH + timedelta(days=days)


@dataclass
class Chunk:
    """In-memory chunk. Arrays are ``(size, size)`` ``uint16`` indexed ``[y, x]``."""

    cx: int
    cy: int
    size: int = DEFAULT_CHUNK_SIZE
    terrain: np.ndarray = field(default=None)  # type: ignore[assignment]
    overlay: np.ndarray = field(default=None)  # type: ignore[assignment]
    objects: list[ChunkObject] = field(default_factory=list)
    period_start: date | None = None
    period_end: date | None = None

    def __post_init__(self) -> None:
        if self.terrain is None:
            self.terrain = np.zeros((self.size, self.size), dtype=np.uint16)
        if self.overlay is None:
            self.overlay = np.zeros((self.size, self.size), dtype=np.uint16)
        self.terrain = np.ascontiguousarray(self.terrain, dtype=np.uint16)
        self.overlay = np.ascontiguousarray(self.overlay, dtype=np.uint16)
        self.objects = [ChunkObject(*o) for o in self.objects]
        self.validate()

    def validate(self) -> None:
        if not (0 < self.size <= 0xFFFF):
            raise ChunkFormatError(f"chunk_size out of u16 range: {self.size}")
        if self.size > 256:
            # cell_x/cell_y are u8 in the objects layer, so larger chunks cannot address objects.
            raise ChunkFormatError(f"chunk_size {self.size} > 256: object cells are u8")
        shape = (self.size, self.size)
        if self.terrain.shape != shape:
            raise ChunkFormatError(f"terrain shape {self.terrain.shape} != {shape}")
        if self.overlay.shape != shape:
            raise ChunkFormatError(f"overlay shape {self.overlay.shape} != {shape}")
        for o in self.objects:
            if not (0 <= o.cell_x < self.size and 0 <= o.cell_y < self.size):
                raise ChunkFormatError(f"object cell {o.cell_x},{o.cell_y} outside chunk")
            if not (0 <= o.tile_id <= 0xFFFF and 0 <= o.flags <= 0xFFFF):
                raise ChunkFormatError(f"object tile_id/flags out of u16 range: {o}")
        if (self.period_start is None) != (self.period_end is None):
            raise ChunkFormatError("period_start and period_end must both be set or both None")

    @property
    def has_objects(self) -> bool:
        return len(self.objects) > 0

    @property
    def layer_count(self) -> int:
        return 3 if self.has_objects else 2


def encode(chunk: Chunk, *, zstd_level: int = 3) -> bytes:
    """Serialize a :class:`Chunk` to ``.bmc`` bytes."""
    chunk.validate()
    parts = [
        chunk.terrain.astype("<u2", copy=False).tobytes(order="C"),
        chunk.overlay.astype("<u2", copy=False).tobytes(order="C"),
    ]
    flags = 0
    if chunk.has_objects:
        flags |= FLAG_HAS_OBJECTS
        parts.append(struct.pack("<I", len(chunk.objects)))
        parts.extend(OBJECT_STRUCT.pack(o.cell_x, o.cell_y, o.tile_id, o.flags, 0) for o in chunk.objects)
    raw = b"".join(parts)
    payload = zstandard.ZstdCompressor(level=zstd_level, write_content_size=True).compress(raw)
    header = HEADER_STRUCT.pack(
        MAGIC,
        SPEC_VERSION,
        flags,
        chunk.size,
        chunk.layer_count,
        0,
        chunk.cx,
        chunk.cy,
        date_to_days(chunk.period_start),
        date_to_days(chunk.period_end),
        len(payload),
    )
    return header + payload


@dataclass(frozen=True)
class ChunkHeader:
    """Parsed 32-byte header, for inspection without decompressing the payload."""

    spec_version: int
    flags: int
    chunk_size: int
    layer_count: int
    reserved: int
    cx: int
    cy: int
    period_start: int
    period_end: int
    payload_len: int

    @property
    def has_objects(self) -> bool:
        return bool(self.flags & FLAG_HAS_OBJECTS)


def decode_header(data: bytes) -> ChunkHeader:
    if len(data) < HEADER_SIZE:
        raise ChunkFormatError(f"file too short for header: {len(data)} bytes")
    magic, spec, flags, size, layers, reserved, cx, cy, ps, pe, plen = HEADER_STRUCT.unpack_from(data, 0)
    if magic != MAGIC:
        raise ChunkFormatError(f"bad magic {magic!r}, expected {MAGIC!r}")
    if spec != SPEC_VERSION:
        raise ChunkFormatError(f"unsupported spec_version {spec}, this codec speaks {SPEC_VERSION}")
    has_objects = bool(flags & FLAG_HAS_OBJECTS)
    if layers != (3 if has_objects else 2):
        raise ChunkFormatError(f"layer_count {layers} inconsistent with flags {flags:#06x}")
    if size == 0:
        raise ChunkFormatError("chunk_size is 0")
    return ChunkHeader(spec, flags, size, layers, reserved, cx, cy, ps, pe, plen)


def decode(data: bytes) -> Chunk:
    """Parse ``.bmc`` bytes into a :class:`Chunk`."""
    h = decode_header(data)
    if len(data) != HEADER_SIZE + h.payload_len:
        raise ChunkFormatError(
            f"file length {len(data)} != header(32) + payload_len({h.payload_len})"
        )
    raw = zstandard.ZstdDecompressor().decompress(
        data[HEADER_SIZE:], max_output_size=64 * 1024 * 1024
    )
    n = h.chunk_size * h.chunk_size
    layer_bytes = n * 2
    offset = 0
    terrain = np.frombuffer(raw, dtype="<u2", count=n, offset=offset).reshape(h.chunk_size, h.chunk_size)
    offset += layer_bytes
    overlay = np.frombuffer(raw, dtype="<u2", count=n, offset=offset).reshape(h.chunk_size, h.chunk_size)
    offset += layer_bytes
    objects: list[ChunkObject] = []
    if h.has_objects:
        if len(raw) < offset + 4:
            raise ChunkFormatError("payload truncated before objects count")
        (count,) = struct.unpack_from("<I", raw, offset)
        offset += 4
        end = offset + count * OBJECT_STRUCT.size
        if len(raw) < end:
            raise ChunkFormatError(f"payload truncated: {count} objects declared")
        recs = np.frombuffer(raw, dtype=OBJECT_DTYPE, count=count, offset=offset)
        objects = [
            ChunkObject(int(r["cell_x"]), int(r["cell_y"]), int(r["tile_id"]), int(r["flags"]))
            for r in recs
        ]
        offset = end
    if offset != len(raw):
        raise ChunkFormatError(f"{len(raw) - offset} trailing bytes in payload")
    return Chunk(
        cx=h.cx,
        cy=h.cy,
        size=h.chunk_size,
        terrain=terrain.copy(),
        overlay=overlay.copy(),
        objects=objects,
        period_start=days_to_date(h.period_start),
        period_end=days_to_date(h.period_end),
    )


def chunk_filename(cx: int, cy: int) -> str:
    """``<cx>_<cy>.bmc`` as laid out in chunk.md."""
    return f"{cx}_{cy}.bmc"


def write_chunk(path: str | Path, chunk: Chunk) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode(chunk))
    return path


def read_chunk(path: str | Path) -> Chunk:
    return decode(Path(path).read_bytes())


def read_header(path: str | Path) -> ChunkHeader:
    with open(path, "rb") as f:
        return decode_header(f.read(HEADER_SIZE))
