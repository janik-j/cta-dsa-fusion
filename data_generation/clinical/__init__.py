"""Clinical DSA data generation pipeline.

Converts real clinical DSA DICOM acquisitions into the training format used by
the 3D vessel reconstruction pipeline. Requires upstream DiffDRR camera pose
registration (``parameters.pt`` files) as a prerequisite.

Quick start::

    python -m data_generation.clinical.cli \\
        --patient-id case_1 \\
        --artery lica --views ap lat \\
        --mip-offset-start 3 --mip-offset-end 15

See ``docs/clinical-data.html`` for the full guide.

Module layout::

    data_generation.clinical.cli          CLI entry point
    data_generation.clinical.config       ClinicalConfig dataclass
    data_generation.clinical.pipeline     ClinicalPipeline orchestrator
    data_generation.clinical.loaders      DICOM, segmentation, prior loaders
    data_generation.clinical.processors   DSA computation, frame filtering
    data_generation.clinical.exporters    Dataset + residual export
    data_generation.clinical.utils        Coordinate transforms, image/mesh I/O
"""
