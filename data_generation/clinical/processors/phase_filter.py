"""Frame-offset filtering for DSA MIP computation.

Selects a subset of fill (contrast) frames based on start/end offsets
relative to the first fill frame.  This makes the selection consistent
across patients regardless of when contrast injection starts.

Example
-------
With ``offset_start=3, offset_end=15`` and ``first_fill_frame=4``,
the selected frame range is ``[7, 19)`` — i.e. 12 frames starting
3 frames after the first contrast appearance.
"""

from __future__ import annotations


class FrameFilter:
    """Select DSA fill frames by offset from the first fill frame."""

    def __init__(self, offset_start: int = 3, offset_end: int = 15) -> None:
        self.offset_start = int(offset_start)
        self.offset_end = int(offset_end)

    def filter_frames(self, fill_frames: list[int]) -> list[int]:
        """Return the subset of *fill_frames* within the configured offset range.

        If ``offset_end`` is 0, all frames from ``offset_start`` onward are
        included (open-ended).
        """
        if not fill_frames:
            return []

        first = min(int(f) for f in fill_frames)
        abs_start = first + self.offset_start

        if self.offset_end <= 0:
            return [int(f) for f in fill_frames if int(f) >= abs_start]

        abs_end = first + self.offset_end
        return [int(f) for f in fill_frames if abs_start <= int(f) < abs_end]

    @property
    def is_full_sequence(self) -> bool:
        """Whether this filter passes all frames (start=0, end=0)."""
        return self.offset_start == 0 and self.offset_end <= 0

    def __repr__(self) -> str:
        end = "end" if self.offset_end <= 0 else str(self.offset_end)
        return f"FrameFilter(+{self.offset_start}..+{end})"
