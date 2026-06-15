"""Waters SONAR build-from-parameters schedule synthesis.

These tests exercise the pure-Python ``build_sonar_schedule`` (no connector / no DB)
and the instrument registration. The full builder (which writes to a DB) is covered
by the simulator smoke path, not here.
"""

import math

import pytest

from imspy_simulation.timsim.jobs.waters_acquisition import build_sonar_schedule
from imspy_simulation.timsim.jobs.register_prediction_set import (
    is_waters_instrument,
    resolve_instrument_activation,
)


def _schedule(**kw):
    params = dict(
        mz_start=400.0, mz_end=900.0, window_width=20.0, window_step=20.0,
        cycle_time_s=0.5, gradient_length_s=5.0,
        ce_intercept=5.0, ce_slope_per_mz=0.04,
    )
    params.update(kw)
    return build_sonar_schedule(**params)


def test_contiguous_windows_tile_the_range():
    _, centers, _ = _schedule()
    # 400-900 / 20 Da contiguous -> 25 windows, centered 410..890.
    assert len(centers) == 25
    assert centers[0] == 410.0
    assert centers[-1] == 890.0
    # contiguous: consecutive centers differ by exactly the width.
    assert all(round(b - a, 6) == 20.0 for a, b in zip(centers, centers[1:]))


def test_overlapping_step_increases_window_count():
    # step 2.5 Da (the faithful SONAR quad scan) -> many more, overlapping windows.
    _, centers, _ = _schedule(window_step=2.5)
    assert len(centers) > 150  # ~193 for 400-900/20@2.5
    # first window still starts at the range floor.
    assert centers[0] == 410.0


def test_each_cycle_is_ms1_then_windows_with_rolling_ce():
    schedule, centers, n_cycles = _schedule(gradient_length_s=1.0)  # 2 cycles @ 0.5s
    assert n_cycles == 2
    # one cycle = 1 MS1 + len(centers) MS2.
    per_cycle = 1 + len(centers)
    assert len(schedule) == per_cycle * n_cycles
    # first row of each cycle is MS1 (ms_level 1, no window/CE).
    for c in range(n_cycles):
        scan, rt, ms_level, center, width, ce = schedule[c * per_cycle]
        assert ms_level == 1 and center is None and width is None and ce is None
    # an MS2 row carries the window + rolling CE = intercept + slope*center.
    scan, rt, ms_level, center, width, ce = schedule[1]
    assert ms_level == 2 and width == 20.0
    assert math.isclose(ce, 5.0 + 0.04 * center)
    # retention times are non-decreasing.
    times = [r[1] for r in schedule]
    assert times == sorted(times)


def test_non_divisible_span_is_fully_covered_via_ceil():
    # span 510 (400-910) is NOT divisible by 20: ceil must add a window so the last
    # window's upper edge reaches/exceeds mz_end (full coverage, slight overshoot allowed).
    _, centers, _ = _schedule(mz_end=910.0)  # span 510, width/step 20
    last_upper = centers[-1] + 20.0 / 2.0
    assert last_upper >= 910.0  # no gap at the top of the range
    # exactly-divisible span must NOT overshoot (float-noise epsilon guards this).
    _, centers2, _ = _schedule()  # span 500
    assert centers2[-1] + 10.0 == 900.0
    assert len(centers2) == 25


def test_invalid_geometry_raises():
    with pytest.raises(ValueError):
        _schedule(mz_end=300.0)  # end <= start
    with pytest.raises(ValueError):
        _schedule(window_width=0.0)
    with pytest.raises(ValueError):
        _schedule(cycle_time_s=0.0)


def test_degenerate_geometry_rejected():
    # window wider than the scanned span.
    with pytest.raises(ValueError, match="span"):
        _schedule(window_width=600.0)  # span is 500
    # step larger than width would leave gaps.
    with pytest.raises(ValueError, match="gap"):
        _schedule(window_width=20.0, window_step=25.0)
    # non-positive / non-finite gradient.
    with pytest.raises(ValueError):
        _schedule(gradient_length_s=0.0)
    with pytest.raises(ValueError):
        _schedule(gradient_length_s=float("inf"))


def test_non_positive_rolling_ce_rejected():
    # intercept/slope that drive CE <= 0 over the window range must be rejected.
    with pytest.raises(ValueError, match="collision energy"):
        _schedule(ce_intercept=-100.0, ce_slope_per_mz=0.0)


def test_instrument_registered_as_waters_hcd_nce():
    assert is_waters_instrument("waters_synapt_xs")
    assert is_waters_instrument("WATERS_SYNAPT_XS")  # case-insensitive
    assert not is_waters_instrument("bruker_timstof")
    method, unit = resolve_instrument_activation("waters_synapt_xs")
    assert method == "hcd" and unit == "nce"
