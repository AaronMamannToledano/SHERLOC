"""Minimal smoke tests so CI has something to execute.

These tests deliberately do NOT depend on the real ImPower150 or MSK-CHORD
data. They only verify that the package is importable and that the encoders
can be instantiated. Extend `tests/` with shape/dtype checks on dummy tensors
as the project grows.
"""
from __future__ import annotations


def test_package_imports():
    import sherloc  # noqa: F401


def test_encoders_importable():
    from sherloc.ctDNA_Dynamics_Encoder import SHERLOC  # noqa: F401
    from sherloc.ctDNA_Dynamics_Encoder import GRU  # noqa: F401
    from sherloc.ctDNA_Dynamics_Encoder import RNN  # noqa: F401
    from sherloc.ctDNA_Dynamics_Encoder import MLP_DeepSurv  # noqa: F401
    from sherloc.ctDNA_Dynamics_Encoder import Featurewise_Transformer  # noqa: F401
    from sherloc.ctDNA_Dynamics_Encoder import Temporal_Transformer  # noqa: F401
    from sherloc.Genomic_Interaction_Encoder import MLP_SolidBiopsy  # noqa: F401
