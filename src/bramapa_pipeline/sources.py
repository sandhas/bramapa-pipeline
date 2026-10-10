"""Public geodata sources for the real level 1 and an on-disk download cache.

Everything here is public and needs no login:

* **Natural Earth 10m** (public domain): countries, land boundary lines, rivers, lakes, roads.
* **IBGE** (open data): 2022 state (UF) polygons and the 1:250k biome vectors.
* **NOAA ETOPO 2022** (public domain): a coarse elevation raster exported through the NCEI
  ImageServer for the South America bounding box. Optional: it only paints ``montanha``.
* **IBGE municipal meshes** per UF (levels 2 and 3), see :func:`municipios_source`.
* **MapBiomas** coverage COGs (CC-BY-SA 4.0), read remotely by window in ``mapbiomas.py``.
* **OpenStreetMap** via Overpass (ODbL), roads/railways/rivers in ``osm.py``.

Downloads land in ``data/raw/`` (gitignored) and are skipped when the file already exists,
whatever its content (no checksum). Delete a file to force a re-download.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import requests

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data/raw")

NE_CDN = "https://naciscdn.org/naturalearth/10m"
NE_S3 = "https://naturalearth.s3.amazonaws.com"
IBGE_GEOFTP = "https://geoftp.ibge.gov.br"

# NOAA ETOPO 2022 via the ArcGIS ImageServer export: lon -82..-30, lat -36..13 at ~2.2 km/px.
ETOPO_BBOX = (-82.0, -36.0, -30.0, 13.0)
ETOPO_SIZE = (2400, 2260)
ETOPO_URL = (
    "https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_all/ImageServer/exportImage"
    f"?bbox={ETOPO_BBOX[0]},{ETOPO_BBOX[1]},{ETOPO_BBOX[2]},{ETOPO_BBOX[3]}&bboxSR=4326"
    f"&size={ETOPO_SIZE[0]},{ETOPO_SIZE[1]}&imageSR=4326&format=tiff&pixelType=F32&f=image"
)


@dataclass(frozen=True)
class Source:
    key: str
    """Short name used by the CLI and by :func:`path_for`."""
    filename: str
    urls: tuple[str, ...]
    """Tried in order; the first that answers wins."""
    licence: str
    optional: bool = False
    """Optional sources (elevation) only log a warning when they cannot be fetched."""
    min_bytes: int = 1024
    """A response smaller than this is treated as an error page, not data."""
    description: str = ""
    layer: str | None = None
    """For shapefile zips: the layer name pyogrio should open (defaults to the zip's only layer)."""
    fields: tuple[str, ...] = field(default=())
    """Attribute columns the builder needs; everything else is dropped on read."""


def _ne(name: str, theme: str, fields: tuple[str, ...], description: str) -> Source:
    return Source(
        key=name,
        filename=f"{name}.zip",
        urls=(f"{NE_CDN}/{theme}/{name}.zip", f"{NE_S3}/10m_{theme}/{name}.zip"),
        licence="Natural Earth, public domain",
        description=description,
        fields=fields,
    )


SOURCES: dict[str, Source] = {
    s.key: s
    for s in (
        _ne(
            "ne_10m_admin_0_countries",
            "cultural",
            ("ISO_A2", "ISO_A2_EH", "ADM0_A3", "NAME", "NAME_PT"),
            "country polygons (land mask + pais:XX entities)",
        ),
        _ne("ne_10m_admin_0_boundary_lines_land", "cultural", (), "international land boundaries"),
        _ne("ne_10m_rivers_lake_centerlines", "physical", ("scalerank",), "rivers"),
        _ne("ne_10m_lakes", "physical", ("scalerank",), "lakes"),
        _ne("ne_10m_roads", "cultural", ("type",), "major roads"),
        Source(
            key="ibge_uf_2022",
            filename="BR_UF_2022.zip",
            urls=(
                f"{IBGE_GEOFTP}/organizacao_do_territorio/malhas_territoriais/malhas_municipais/"
                "municipio_2022/Brasil/BR/BR_UF_2022.zip",
            ),
            licence="IBGE, dados abertos",
            description="Brazilian states (CD_UF, NM_UF, SIGLA_UF)",
            fields=("CD_UF", "NM_UF", "SIGLA_UF"),
        ),
        Source(
            key="ibge_biomas",
            filename="Biomas_250mil.zip",
            urls=(f"{IBGE_GEOFTP}/informacoes_ambientais/estudos_ambientais/biomas/vetores/Biomas_250mil.zip",),
            licence="IBGE, dados abertos",
            description="Brazilian biomes 1:250k (Bioma)",
            fields=("Bioma",),
        ),
        Source(
            key="etopo2022",
            filename="etopo2022_brazil.tif",
            urls=(ETOPO_URL,),
            licence="NOAA ETOPO 2022, public domain",
            optional=True,
            min_bytes=100_000,
            description="coarse elevation (montanha above 2000 m)",
        ),
    )
}

REQUIRED_KEYS = tuple(k for k, s in SOURCES.items() if not s.optional)


def municipios_source(uf_code: int) -> Source:
    """IBGE 2022 municipal mesh of one UF (``CD_MUN``, ``NM_MUN``), ~8 MB for RS. Not in
    :data:`SOURCES` so ``fetch`` without ``--only`` never downloads all 27 states."""
    from .level import UF_SIGLAS

    sigla = UF_SIGLAS[int(uf_code)]
    return Source(
        key=f"ibge_municipios_{sigla}",
        filename=f"{sigla}_Municipios_2022.zip",
        urls=(
            f"{IBGE_GEOFTP}/organizacao_do_territorio/malhas_territoriais/malhas_municipais/"
            f"municipio_2022/UFs/{sigla}/{sigla}_Municipios_2022.zip",
        ),
        licence="IBGE, dados abertos",
        description=f"municipalities of {sigla} (CD_MUN, NM_MUN)",
        fields=("CD_MUN", "NM_MUN", "SIGLA_UF"),
    )


# MapBiomas Brazil land use / land cover, 30 m, EPSG:4326, one Cloud-Optimised GeoTIFF per year in
# the public GCS bucket (CC-BY-SA 4.0). Read remotely through ``/vsicurl/`` (see ``mapbiomas.py``);
# the whole files are 700-800 MB each and are never downloaded. Newest first. Only collection 9
# ships internal overviews: the others can only be read at native 30 m without pulling the full
# window, so decimated (level 2) reads skip them.
MAPBIOMAS_BUCKET = "https://storage.googleapis.com/mapbiomas-public/initiatives/brasil"
MAPBIOMAS_COVERAGES: tuple[tuple[int, int, str], ...] = (
    (11, 2025, f"{MAPBIOMAS_BUCKET}/collection11/lulc/coverage/brazil_coverage/brazil_coverage-col11_2025.tif"),
    (10, 2024, f"{MAPBIOMAS_BUCKET}/collection_10/lulc/coverage/brazil_coverage_2024.tif"),
    (9, 2023, f"{MAPBIOMAS_BUCKET}/collection_9/lclu/coverage/brasil_coverage_2023.tif"),
    (8, 2022, f"{MAPBIOMAS_BUCKET}/collection_8/lclu/coverage/brasil_coverage_2022.tif"),
)
"""``(collection, year, url)`` candidates, tried in order."""
MAPBIOMAS_LICENCE = "MapBiomas, CC-BY-SA 4.0"

# OpenStreetMap via the Overpass API (data ODbL; attribution "© OpenStreetMap contributors").
OVERPASS_ENDPOINTS: tuple[str, ...] = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
OSM_LICENCE = "OpenStreetMap contributors, ODbL 1.0"


class SourceError(RuntimeError):
    """A required download failed on every URL."""


def path_for(source: Source | str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    src = SOURCES[source] if isinstance(source, str) else source
    return Path(cache_dir) / src.filename


def _download(url: str, dest: Path, *, timeout: float, min_bytes: int) -> int:
    """Stream ``url`` into ``dest`` through a temp file; returns the byte count."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": "bramapa-pipeline/0.1"}) as r:
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        tmp = tempfile.NamedTemporaryFile(dir=dest.parent, prefix=dest.name + ".", suffix=".part", delete=False)
        try:
            with tmp:
                n = 0
                for block in r.iter_content(chunk_size=1 << 20):
                    tmp.write(block)
                    n += len(block)
            if n < min_bytes:
                head = Path(tmp.name).read_bytes()[:200]
                raise SourceError(f"{url}: only {n} bytes (Content-Type {ctype!r}); body starts {head!r}")
            if "html" in ctype.lower() or "json" in ctype.lower():
                head = Path(tmp.name).read_bytes()[:200]
                raise SourceError(f"{url}: server answered {ctype!r} instead of data; body starts {head!r}")
            shutil.move(tmp.name, dest)
            dest.chmod(0o644)
        finally:
            Path(tmp.name).unlink(missing_ok=True)
    return n


def fetch(
    source: Source | str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    *,
    timeout: float = 120.0,
    force: bool = False,
) -> Path | None:
    """Return the cached file for ``source``, downloading it if missing.

    Required sources raise :class:`SourceError` when every URL fails; optional ones return
    ``None`` after logging a warning.
    """
    src = SOURCES[source] if isinstance(source, str) else source
    dest = path_for(src, cache_dir)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        log.info("%s: cached (%d bytes)", dest, dest.stat().st_size)
        return dest
    errors: list[str] = []
    for url in src.urls:
        try:
            log.info("%s: downloading %s", src.key, url)
            n = _download(url, dest, timeout=timeout, min_bytes=src.min_bytes)
            log.info("%s: %d bytes -> %s", src.key, n, dest)
            return dest
        except (requests.RequestException, SourceError, OSError) as e:
            errors.append(f"{url}: {type(e).__name__}: {e}")
            log.warning("%s: %s", src.key, errors[-1])
    msg = f"{src.key}: all downloads failed:\n  " + "\n  ".join(errors)
    if src.optional:
        log.warning("%s (optional source, continuing without it)", msg)
        return None
    raise SourceError(msg)


def fetch_all(
    cache_dir: Path = DEFAULT_CACHE_DIR, *, keys: tuple[str, ...] | None = None, timeout: float = 120.0, force: bool = False
) -> dict[str, Path | None]:
    """Fetch every source (or ``keys``); returns ``key -> path`` (``None`` for skipped optionals)."""
    out: dict[str, Path | None] = {}
    for key in keys or tuple(SOURCES):
        out[key] = fetch(key, cache_dir, timeout=timeout, force=force)
    return out


def cached_status(cache_dir: Path = DEFAULT_CACHE_DIR) -> list[tuple[Source, Path, int | None]]:
    """``(source, path, size or None)`` for every source, without touching the network."""
    rows = []
    for src in SOURCES.values():
        p = path_for(src, cache_dir)
        rows.append((src, p, p.stat().st_size if p.exists() else None))
    return rows


__all__ = [
    "DEFAULT_CACHE_DIR",
    "MAPBIOMAS_COVERAGES",
    "MAPBIOMAS_LICENCE",
    "OSM_LICENCE",
    "OVERPASS_ENDPOINTS",
    "municipios_source",
    "ETOPO_BBOX",
    "ETOPO_SIZE",
    "REQUIRED_KEYS",
    "SOURCES",
    "Source",
    "SourceError",
    "cached_status",
    "fetch",
    "fetch_all",
    "path_for",
]
