"""Bramapa offline pipeline: turns geodata into tile chunks (.bmc) and level manifests.

Fase 0: codec for the chunk format and a synthetic level generator. See the spec at
``bramapa/docs/spec/chunk.md`` for the contract this package implements.
"""

SPEC_VERSION = 0

__all__ = ["SPEC_VERSION"]
