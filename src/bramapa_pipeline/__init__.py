"""Bramapa offline pipeline: turns geodata into tile chunks (.bmc) and level manifests.

Fase 0/1: codec for the chunk format, autotiling and a synthetic level generator. Fase 2: the
real level 1 (Brazil and neighbours) from Natural Earth, IBGE and ETOPO (``sources``, ``level1``).
Fase 4: levels 2 (state) and 3 (municipality) from MapBiomas, IBGE municipal meshes and OSM
(``mapbiomas``, ``osm``, ``region``, ``level2``, ``level3``), written under ``<root>/<level>/<entity>/``.
See the spec at ``bramapa/docs/spec/chunk.md`` for the contract this package implements.
"""

SPEC_VERSION = 0

__all__ = ["SPEC_VERSION"]
