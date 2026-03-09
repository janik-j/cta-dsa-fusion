"""
TopBrain-based Dataset Generation Pipeline

This module provides tools for generating this framework training datasets from
TopBrain CTA volumes with simulated projections and vessel priors.

Main Components
---------------
processors/cta.py
    CTA processing utilities: volume loading, TIGRE forward projection,
    DSA simulation, baseline/contrast volume creation.

processors/geometry.py
    Geometry generation: PLY point clouds from segmentations, mesh export,
    visual hull computation, coordinate conversion.

loaders/cases.py
    Case discovery utilities and view transform creation.

exporters/views.py / exporters/priors.py / exporters/residuals.py
    Product modules for view projections, prior PLYs, and residual PLYs.

Usage
-----
Generate complete dataset for a case:

    python -m data_generation.topbrain.cli --case topcow_ct_001

Generate all cases in batch:

    python -m data_generation.topbrain.cli --batch
"""

__all__: list[str] = []
