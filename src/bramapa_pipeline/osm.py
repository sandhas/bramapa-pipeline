"""OpenStreetMap lines (roads, railways, rivers) through the Overpass API, with a JSON cache.

Queries are plain OverpassQL with ``out geom;`` so every way carries its own coordinates (no
node lookups). :func:`fetch_overpass` rotates through the public endpoints with retries, treats the
"server too busy" HTML answers as failures and caches the JSON in ``data/raw/``; :func:`parse_lines`
turns the JSON into ``(tags, LineString)`` pairs and :func:`classify_way` maps tags to overlay covers.

Licence: OSM data is ODbL 1.0; anything derived from it must credit "© OpenStreetMap contributors"
and, if redistributed as a database, stay under ODbL. See ``README.md``.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import requests

from .sources import OVERPASS_ENDPOINTS

log = logging.getLogger(__name__)

USER_AGENT = "bramapa-pipeline/0.1 (+https://github.com/sandhas/bramapa-pipeline)"

ROAD_CLASSES_LEVEL2 = ("motorway", "trunk", "primary")
ROAD_CLASSES_LEVEL3 = ("motorway", "trunk", "primary", "secondary")
ROAD_CLASSES_MINOR = ("tertiary", "residential")
MINOR_ROADS_MAX_WAYS = 60_000
"""Above this many ``tertiary``/``residential`` ways the level 3 build skips them (too dense to draw)."""


class OverpassError(RuntimeError):
    """Every endpoint failed (or answered an error page)."""


def bbox_filter(bbox: tuple[float, float, float, float]) -> str:
    """Lon/lat ``(w, s, e, n)`` -> Overpass ``(s,w,n,e)`` bbox filter."""
    w, s, e, n = bbox
    return f"({s:.6f},{w:.6f},{n:.6f},{e:.6f})"


def highway_selector(classes: tuple[str, ...]) -> str:
    return 'way[highway~"^(' + "|".join(classes) + ')$"]'


RAIL_SELECTOR = "way[railway=rail]"
RIVER_SELECTOR = "way[waterway=river]"


def build_query(bbox: tuple[float, float, float, float], selectors: tuple[str, ...], *, out: str = "geom", timeout: int = 180) -> str:
    """``[out:json][timeout:T];(sel1(bbox);sel2(bbox););out geom;`` (or ``out count;``)."""
    bb = bbox_filter(bbox)
    body = "".join(f"{sel}{bb};" for sel in selectors)
    return f"[out:json][timeout:{timeout}];({body});out {out};"


def _looks_like_error(text: str) -> str | None:
    head = text.lstrip()[:400].lower()
    if head.startswith("<!doctype") or head.startswith("<html") or head.startswith("<?xml"):
        if "too busy" in text.lower() or "timeout" in text.lower():
            return "server too busy / timed out"
        return "HTML error page"
    return None


def fetch_overpass(
    query: str,
    cache_path: Path | None = None,
    *,
    endpoints: tuple[str, ...] = OVERPASS_ENDPOINTS,
    retries: int = 2,
    timeout: float = 240.0,
    backoff: float = 5.0,
) -> dict:
    """POST ``query`` to the first endpoint that answers valid JSON; cache it; return the document.

    ``retries`` is the number of full passes over the endpoints. A cached file is returned as is.
    """
    if cache_path is not None and Path(cache_path).exists():
        doc = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        log.info("overpass: cached %s (%d elements)", cache_path, len(doc.get("elements", [])))
        return doc
    errors: list[str] = []
    for attempt in range(retries + 1):
        for ep in endpoints:
            t0 = time.perf_counter()
            try:
                r = requests.post(ep, data={"data": query}, timeout=timeout, headers={"User-Agent": USER_AGENT})
                why = _looks_like_error(r.text) if r.status_code == 200 or r.text else None
                if r.status_code != 200 or why:
                    errors.append(f"{ep}: HTTP {r.status_code}{' ' + why if why else ''}")
                    log.warning("overpass: %s (%.1fs)", errors[-1], time.perf_counter() - t0)
                    continue
                doc = r.json()
                if "elements" not in doc:
                    errors.append(f"{ep}: no `elements` in answer: {r.text[:120]!r}")
                    continue
                remark = doc.get("remark")
                if remark and "runtime error" in remark:
                    errors.append(f"{ep}: {remark}")
                    log.warning("overpass: %s", errors[-1])
                    continue
                if remark:
                    log.warning("overpass remark: %s", remark)
                log.info("overpass: %s answered %d elements (%d bytes) in %.1fs", ep, len(doc["elements"]), len(r.content), time.perf_counter() - t0)
                if cache_path is not None:
                    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(cache_path).write_bytes(r.content)
                return doc
            except (requests.RequestException, ValueError) as e:
                errors.append(f"{ep}: {type(e).__name__}: {e}")
                log.warning("overpass: %s", errors[-1])
        if attempt < retries:
            wait = backoff * (attempt + 1)
            log.info("overpass: all endpoints failed, retrying in %.0fs", wait)
            time.sleep(wait)
    raise OverpassError("Overpass failed on every endpoint:\n  " + "\n  ".join(errors))


def count_ways(bbox, selectors: tuple[str, ...], **kw) -> int:
    """``out count;`` for the selectors; -1 if the servers refuse the count (older mirrors)."""
    try:
        doc = fetch_overpass(build_query(bbox, selectors, out="count"), None, retries=0, **kw)
    except OverpassError as e:
        log.warning("overpass count failed: %s", e)
        return -1
    for el in doc.get("elements", []):
        if el.get("type") == "count":
            return int(el.get("tags", {}).get("ways", el.get("tags", {}).get("total", -1)))
    return -1


# ----------------------------------------------------------------------------- parsing


def parse_lines(doc: dict) -> list[tuple[dict, list[tuple[float, float]]]]:
    """``out geom`` ways -> ``(tags, [(lon, lat), ...])`` with at least two vertices."""
    out = []
    for el in doc.get("elements", []):
        if el.get("type") != "way":
            continue
        geom = el.get("geometry") or []
        coords = [(float(p["lon"]), float(p["lat"])) for p in geom if "lon" in p and "lat" in p]
        if len(coords) >= 2:
            out.append((el.get("tags") or {}, coords))
    return out


def classify_way(tags: dict, road_classes: tuple[str, ...] = ROAD_CLASSES_LEVEL3 + ROAD_CLASSES_MINOR) -> str | None:
    """Overlay cover for a way: ``rodovia``, ``ferrovia``, ``rio`` or ``None`` (ignored)."""
    if tags.get("railway") == "rail":
        return "ferrovia"
    if tags.get("waterway") == "river":
        return "rio"
    hw = tags.get("highway")
    if hw in road_classes:
        return "rodovia"
    return None


def lines_to_gdf(lines, road_classes: tuple[str, ...], crs: str = "EPSG:4326"):
    """``(tags, coords)`` -> GeoDataFrame(``overlay``, geometry) in lon/lat, unclassified ways dropped."""
    import geopandas as gpd
    from shapely.geometry import LineString

    covers, geoms = [], []
    for tags, coords in lines:
        c = classify_way(tags, road_classes)
        if c is None:
            continue
        covers.append(c)
        geoms.append(LineString(coords))
    return gpd.GeoDataFrame({"overlay": covers, "geometry": geoms}, crs=crs)


__all__ = [
    "MINOR_ROADS_MAX_WAYS",
    "OverpassError",
    "RAIL_SELECTOR",
    "RIVER_SELECTOR",
    "ROAD_CLASSES_LEVEL2",
    "ROAD_CLASSES_LEVEL3",
    "ROAD_CLASSES_MINOR",
    "bbox_filter",
    "build_query",
    "classify_way",
    "count_ways",
    "fetch_overpass",
    "highway_selector",
    "lines_to_gdf",
    "parse_lines",
]
