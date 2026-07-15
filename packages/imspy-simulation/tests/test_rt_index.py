"""RT as a portable index (structure) mapped to seconds per run (measurement).

The claims of the RT port, asserted:

1. **Cross-gradient portability.** The same index, mapped onto a 30-minute and a 2-hour gradient,
   scales proportionally — a peptide keeps its gradient *fraction*. This is the RT analog of
   cross-instrument mobility.

2. **Sample independence.** With a *fixed reference range* (the index span over the whole peptide
   space), a peptide lands at the same gradient fraction no matter what else is in the sample. v1
   used each sample's own min/max, so a peptide's RT moved with its neighbours — this test shows the
   two approaches disagree, and that the reference approach is the stable one.
"""

import numpy as np


def index_to_seconds(idx, ref_lo, ref_hi, gradient):
    """The measurement map used in the simulator: index -> [0, gradient] with a reference range."""
    return (np.asarray(idx, float) - ref_lo) / (ref_hi - ref_lo) * gradient


def test_index_maps_proportionally_across_gradients():
    ref_lo, ref_hi = 0.0, 20.0
    idx = np.array([2.0, 10.0, 18.0])
    rt_30min = index_to_seconds(idx, ref_lo, ref_hi, 30 * 60)
    rt_2h = index_to_seconds(idx, ref_lo, ref_hi, 120 * 60)
    # Same fraction of the gradient on both.
    frac_30 = rt_30min / (30 * 60)
    frac_2h = rt_2h / (120 * 60)
    assert np.allclose(frac_30, frac_2h)
    # And a 4x-longer gradient means 4x the seconds.
    assert np.allclose(rt_2h, 4.0 * rt_30min)


def test_fixed_reference_range_is_sample_independent():
    """A peptide shared between two samples of different composition must land at the SAME gradient
    fraction under a fixed reference range — and would NOT under per-sample min/max."""
    ref_lo, ref_hi = 0.0, 30.0
    gradient = 3600.0
    shared_index = 12.0

    # Two samples that happen to contain the shared peptide but otherwise differ in index spread.
    sample_a = np.array([2.0, shared_index, 25.0])
    sample_b = np.array([10.0, shared_index, 14.0])

    # Reference-range mapping: identical for the shared peptide.
    rt_a_ref = index_to_seconds(shared_index, ref_lo, ref_hi, gradient)
    rt_b_ref = index_to_seconds(shared_index, ref_lo, ref_hi, gradient)
    assert rt_a_ref == rt_b_ref

    # v1-style per-sample mapping: the shared peptide gets DIFFERENT times, because the min/max differ.
    rt_a_pop = index_to_seconds(shared_index, sample_a.min(), sample_a.max(), gradient)
    rt_b_pop = index_to_seconds(shared_index, sample_b.min(), sample_b.max(), gradient)
    assert not np.isclose(rt_a_pop, rt_b_pop), "per-sample mapping should move the shared peptide"


def test_order_is_preserved():
    """Whatever the gradient, elution order follows the hydrophobicity index."""
    idx = np.array([5.0, 1.0, 9.0, 3.0])
    rt = index_to_seconds(idx, 0.0, 10.0, 1800.0)
    assert list(np.argsort(rt)) == list(np.argsort(idx))
