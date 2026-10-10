"""Placeholder tile ids shared across the Bramapa repos (spec v0).

These numbers are a provisional convention until ``bramapa-tileset`` publishes a real
``tiles.toml``; the pipeline will then resolve covers to ids through that file instead.
``0`` is always "empty".
"""

from __future__ import annotations

EMPTY = 0

# --- terrain layer ---------------------------------------------------------
CAMPO = 1
AGUA = 2
OCEANO = 3
FLORESTA = 4
AREIA = 5
URBANO = 6
MONTANHA = 7
SAVANA = 8
AGRICULTURA = 9
PASTAGEM = 10
CAATINGA = 11
PANTANAL = 12

# --- overlay layer ---------------------------------------------------------
RODOVIA = 20
RIO = 21
LIMITE_ESTADO = 22
LIMITE_MUNICIPIO = 23
LIMITE_PAIS = 24
FERROVIA = 25

# --- objects ---------------------------------------------------------------
ARVORE = 40
PREDIO = 41
CASA = 42

TERRAIN_NAMES: dict[int, str] = {
    CAMPO: "campo",
    AGUA: "agua",
    OCEANO: "oceano",
    FLORESTA: "floresta",
    AREIA: "areia",
    URBANO: "urbano",
    MONTANHA: "montanha",
    SAVANA: "savana",
    AGRICULTURA: "agricultura",
    PASTAGEM: "pastagem",
    CAATINGA: "caatinga",
    PANTANAL: "pantanal",
}

OVERLAY_NAMES: dict[int, str] = {
    RODOVIA: "rodovia",
    RIO: "rio",
    LIMITE_ESTADO: "limite_estado",
    LIMITE_MUNICIPIO: "limite_municipio",
    LIMITE_PAIS: "limite_pais",
    FERROVIA: "ferrovia",
}

OBJECT_NAMES: dict[int, str] = {
    ARVORE: "arvore",
    PREDIO: "predio",
    CASA: "casa",
}

ALL_NAMES: dict[int, str] = {EMPTY: "vazio", **TERRAIN_NAMES, **OVERLAY_NAMES, **OBJECT_NAMES}


def tile_name(tile_id: int) -> str:
    """Human-readable name for a placeholder tile id (``"?<id>"`` if unknown)."""
    return ALL_NAMES.get(tile_id, f"?{tile_id}")

ALL_NAMES_INV: dict[str, int] = {name: tile_id for tile_id, name in ALL_NAMES.items()}
"""Inverse of :data:`ALL_NAMES` (``"vazio"`` -> 0)."""
