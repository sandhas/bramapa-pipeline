"""MapBiomas land cover: legend → covers, remote windowed reads and the warp onto the rotated grid.

The Brazil coverage is one 30 m ``uint8`` GeoTIFF per year (~155k × 150k px, 700–800 MB) in a public
Google Cloud Storage bucket. :func:`read_coverage` opens it through GDAL's ``/vsicurl/`` and reads
only the window of the level's bbox (HTTP range requests), decimated through the file's internal
overviews when the grid cells are much coarser than 30 m; the result is cached as a small local
GeoTIFF so reruns are offline. :func:`warp_classes` then resamples the classes (mode) straight into
the rotated ``EPSG:5880`` grid, like ``level1`` does for elevation, and :func:`classify` turns
MapBiomas class codes into the pipeline's terrain codes.

Licence: MapBiomas Collection data is CC-BY-SA 4.0 ("Projeto MapBiomas – Coleção N da Série Anual
de Mapas de Cobertura e Uso da Terra do Brasil").
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rasterio.transform import Affine

from .level1 import AGRICULTURA, AGUA, AREIA, CAMPO, FLORESTA, MONTANHA, OCEANO, PANTANAL, PASTAGEM, SAVANA, URBANO, GridSpec
from .sources import MAPBIOMAS_COVERAGES

log = logging.getLogger(__name__)

NODATA_CLASS = 0
"""MapBiomas writes 0 outside Brazil (and in the few unobserved pixels)."""

# MapBiomas legend (collections 8–11 share these codes) -> terrain code. Parent classes (1, 10,
# 14, 18, 22, 26) never appear in the coverage rasters but are mapped for robustness.
MAPBIOMAS_COVER: dict[int, int] = {
    # 1. Forest
    1: FLORESTA,
    3: FLORESTA,  # forest formation
    4: SAVANA,  # savanna formation
    5: PANTANAL,  # mangrove
    6: FLORESTA,  # floodable forest
    49: FLORESTA,  # wooded sandbank vegetation (restinga)
    # 2. Herbaceous and shrubby vegetation
    10: CAMPO,
    11: PANTANAL,  # wetland
    12: CAMPO,  # grassland
    32: PANTANAL,  # hypersaline tidal flat
    29: MONTANHA,  # rocky outcrop
    50: CAMPO,  # herbaceous sandbank vegetation
    13: CAMPO,  # other non-forest formations
    # 3. Farming
    14: PASTAGEM,
    15: PASTAGEM,  # pasture
    18: AGRICULTURA,
    19: AGRICULTURA,  # temporary crop
    39: AGRICULTURA,  # soybean
    20: AGRICULTURA,  # sugar cane
    40: AGRICULTURA,  # rice
    62: AGRICULTURA,  # cotton
    41: AGRICULTURA,  # other temporary crops
    36: AGRICULTURA,  # perennial crop
    46: AGRICULTURA,  # coffee
    47: AGRICULTURA,  # citrus
    35: AGRICULTURA,  # palm oil
    48: AGRICULTURA,  # other perennial crops
    9: FLORESTA,  # forest plantation
    21: PASTAGEM,  # mosaic of uses
    # 4. Non vegetated area
    22: CAMPO,
    23: AREIA,  # beach, dune and sand spot
    24: URBANO,  # urban area
    30: CAMPO,  # mining
    25: CAMPO,  # other non-vegetated areas
    # 5. Water
    26: AGUA,
    33: AGUA,  # river, lake and ocean
    31: AGUA,  # aquaculture
    # 6. Not observed
    27: CAMPO,
}

LEGEND_NAMES: dict[int, str] = {
    0: "nodata", 3: "formacao florestal", 4: "formacao savanica", 5: "mangue", 6: "floresta alagavel",
    9: "silvicultura", 11: "campo alagado", 12: "formacao campestre", 13: "outras nao florestais",
    15: "pastagem", 19: "lavoura temporaria", 20: "cana", 21: "mosaico de usos", 23: "praia e duna",
    24: "area urbanizada", 25: "outras areas nao vegetadas", 27: "nao observado", 29: "afloramento rochoso",
    30: "mineracao", 31: "aquicultura", 32: "apicum", 33: "rio, lago e oceano", 35: "dende", 36: "lavoura perene",
    39: "soja", 40: "arroz", 41: "outras temporarias", 46: "cafe", 47: "citrus", 48: "outras perenes",
    49: "restinga arborea", 50: "restinga herbacea", 62: "algodao",
}  # fmt: skip


def cover_lut(unknown: int = CAMPO) -> np.ndarray:
    """256-entry ``uint8`` table: MapBiomas class -> terrain code; 0 stays 0 (nodata)."""
    lut = np.full(256, unknown, dtype=np.uint8)
    for cls, code in MAPBIOMAS_COVER.items():
        lut[cls] = code
    lut[NODATA_CLASS] = 0
    return lut


def classify(classes: np.ndarray, land: np.ndarray) -> np.ndarray:
    """MapBiomas classes (``uint8``, 0 = nodata) -> terrain codes.

    Nodata becomes ``oceano`` off ``land`` and ``campo`` on it (MapBiomas covers Brazil only, so
    neighbouring countries come out as plain ``campo``); water classes off ``land`` are ``oceano``
    too (the coverage paints coastal waters inside the Brazilian maritime outline as class 33).
    """
    classes = np.asarray(classes, dtype=np.uint8)
    unknown = np.setdiff1d(np.unique(classes), np.array(list(MAPBIOMAS_COVER) + [NODATA_CLASS]))
    if unknown.size:
        log.warning("unknown MapBiomas classes %s painted campo", unknown.tolist())
    terrain = cover_lut()[classes]
    nodata = classes == NODATA_CLASS
    terrain[nodata & land] = CAMPO
    terrain[nodata & ~land] = OCEANO
    terrain[(terrain == AGUA) & ~land] = OCEANO
    return terrain


# ----------------------------------------------------------------------------- remote reads


@dataclass(frozen=True)
class CoverageInfo:
    collection: int
    year: int
    url: str


class MapbiomasError(RuntimeError):
    pass


def _open_candidates(*, need_overviews: bool):
    """Yield ``(CoverageInfo, open dataset)`` for the first candidate that opens (and has overviews
    when asked); the caller closes the dataset."""
    import rasterio

    errors = []
    for collection, year, url in MAPBIOMAS_COVERAGES:
        info = CoverageInfo(collection, year, url)
        try:
            ds = rasterio.open(f"/vsicurl/{url}")
        except Exception as e:  # noqa: BLE001 - try the next candidate
            errors.append(f"{url}: {type(e).__name__}: {e}")
            log.warning("mapbiomas: %s", errors[-1])
            continue
        if need_overviews and not ds.overviews(1):
            log.info("mapbiomas: collection %d (%d) has no overviews, skipping for a decimated read", collection, year)
            ds.close()
            continue
        return info, ds
    raise MapbiomasError("no MapBiomas coverage could be opened:\n  " + "\n  ".join(errors))


def decimation_for(cell_meters: float, source_meters: float = 30.0, oversample: int = 4) -> int:
    """Integer decimation so the read keeps about ``oversample`` source pixels per grid cell side."""
    return max(1, int(math.floor(cell_meters / (oversample * source_meters))))


@dataclass
class Coverage:
    """A windowed MapBiomas read: classes in EPSG:4326 (north-up) plus provenance."""

    classes: np.ndarray
    transform: Affine
    crs: str
    info: CoverageInfo
    from_cache: bool = False


def read_coverage(bbox: tuple[float, float, float, float], cell_meters: float, cache_path: Path) -> Coverage:
    """Read the MapBiomas classes covering lon/lat ``bbox`` (``w, s, e, n``), decimated for ``cell_meters``.

    A previous read cached at ``cache_path`` (GeoTIFF with ``MAPBIOMAS_*`` tags) is reused as is.
    """
    import rasterio
    from rasterio.windows import Window, from_bounds

    cache_path = Path(cache_path)
    if cache_path.exists():
        with rasterio.open(cache_path) as ds:
            tags = ds.tags()
            info = CoverageInfo(int(tags.get("MAPBIOMAS_COLLECTION", 0)), int(tags.get("MAPBIOMAS_YEAR", 0)), tags.get("MAPBIOMAS_URL", ""))
            arr = ds.read(1)
            log.info("mapbiomas: cached %s (%dx%d px, collection %d/%d)", cache_path, arr.shape[1], arr.shape[0], info.collection, info.year)
            return Coverage(arr, ds.transform, ds.crs.to_string(), info, from_cache=True)

    factor = decimation_for(cell_meters)
    t0 = time.perf_counter()
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES"):
        info, ds = _open_candidates(need_overviews=factor > 1)
        with ds:
            w = from_bounds(*bbox, ds.transform).round_offsets().round_lengths()
            w = w.intersection(Window(0, 0, ds.width, ds.height))
            if w.width <= 0 or w.height <= 0:
                raise MapbiomasError(f"bbox {bbox} does not intersect the MapBiomas raster")
            out_shape = (max(1, int(w.height) // factor), max(1, int(w.width) // factor))
            arr = ds.read(1, window=w, out_shape=out_shape)  # nearest, through the overviews
            win_tr = ds.window_transform(w)
            tr = win_tr * Affine.scale(w.width / out_shape[1], w.height / out_shape[0])
            crs = ds.crs.to_string()
    log.info(
        "mapbiomas: collection %d (%d) window %dx%d px read as %dx%d (decimation %d) in %.1fs",
        info.collection, info.year, int(w.width), int(w.height), out_shape[1], out_shape[0], factor, time.perf_counter() - t0,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        cache_path, "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1, dtype="uint8",
        crs=crs, transform=tr, compress="deflate", tiled=True,
    ) as dst:
        dst.write(arr, 1)
        dst.update_tags(MAPBIOMAS_COLLECTION=str(info.collection), MAPBIOMAS_YEAR=str(info.year), MAPBIOMAS_URL=info.url,
                        MAPBIOMAS_BBOX=",".join(f"{v:.5f}" for v in bbox), MAPBIOMAS_DECIMATION=str(factor))
    return Coverage(arr, tr, crs, info)


def warp_classes(cov: Coverage, spec: GridSpec) -> np.ndarray:
    """Resample the classes onto the rotated grid (mode of the source pixels under each cell).

    Cells with no source pixel (outside the window) come out 0 = nodata.
    """
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    t0 = time.perf_counter()
    dst = np.zeros((spec.size, spec.size), dtype=np.uint8)
    reproject(
        source=np.ascontiguousarray(cov.classes, dtype=np.uint8),
        destination=dst,
        src_transform=cov.transform,
        src_crs=cov.crs,
        dst_transform=spec.transform,
        dst_crs=spec.crs,
        resampling=Resampling.mode,
        num_threads=4,
    )
    log.info("mapbiomas: warped %dx%d px -> %dx%d cells in %.1fs", cov.classes.shape[1], cov.classes.shape[0], spec.size, spec.size, time.perf_counter() - t0)
    return dst


__all__ = [
    "Coverage",
    "CoverageInfo",
    "LEGEND_NAMES",
    "MAPBIOMAS_COVER",
    "MapbiomasError",
    "NODATA_CLASS",
    "classify",
    "cover_lut",
    "decimation_for",
    "read_coverage",
    "warp_classes",
]
