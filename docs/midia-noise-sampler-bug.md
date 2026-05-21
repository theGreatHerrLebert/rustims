# Patch: TimSim midiA noise-sampler emits `MsType::Unknown` frames

**Status:** proposed, awaiting your review
**Affected versions:** all current; writer code last touched 2026-01-24 (commit `dddb03fa`, "modular package split") with no functional changes since
**Symptom carrier dataset:** `TIMSIM-MIDIA-150K-24-12.d` (created 2026-05-18)
**Severity:** silently produces non-Bruker-conformant `.d` files; corrupted output passes our own readers but breaks every downstream that hashes by `Frames.Id` (OpenTIMS, ionmaiden's `d2ms1`)
**Salvage available:** [`scripts/repair_timsim_midia_frames.py`](scripts/repair_timsim_midia_frames.py) (one-shot SQLite fixup that re-assigns the corrupted rows their canonical IDs / Times / MsMsTypes; verified end-to-end via opentimspy on the affected `.d`)

---

## Symptom

For midiA acquisitions, the writer emits a number of `Frames` rows that
share **identical corrupted metadata** while carrying real binary
content:

| field      | value           |
|------------|-----------------|
| `Id`       | `1` (duplicated) |
| `Time`     | `0.0`            |
| `MsMsType` | `-1` (`MsType::Unknown`) |
| `TimsId`   | distinct, points to a valid compressed frame in `analysis.tdf_bin` |
| `NumPeaks` | non-zero (32k-48k on the affected dataset) |

On `TIMSIM-MIDIA-150K-24-12.d` this affects **115 of 34,155 frames**
(0.34%). The `Frames.Id` column therefore has 115 duplicates of `Id=1`,
while the canonical MS1-cycle slots they should have occupied
(stride-17 in midiA: 1, 18, 35, 52, ..., 34086) are simply absent. The
SQLite schema treats `Id` as `INTEGER` without `UNIQUE`/`PRIMARY KEY`,
so the corruption passes through `to_sql` writes without complaint.

Our own readers (`rustdf`, `imspy_core`) survive because they index
into the `Frames` row vector by **row position**, not by `Id`.
Bruker's proprietary SDK appears to do the same. OpenTIMS instead
builds an `unordered_map<frame_id, frame_info>` upfront and throws
`IndexError: unordered_map::at` whenever the simulated dataset's
caller asks for one of the missing IDs.

## Root cause

Three components compose into the bug; each is locally defensible.

### 1. Empty-frame fallback in `get_frame` returns `Default::default()`

`rustdf/src/data/handle.rs:692`:
```rust
if raw_frame.scan.is_empty() {
    return TimsFrame::default();
}
```
`TimsFrame::default()` is `{ frame_id: 0, ms_type: MsType::Unknown, ... }`.
Some MS1 frames in the *reference* Bruker run that TimSim is sampling
from are physically empty (zero peaks); for those, `get_frame()`
silently returns a frame that carries neither the requested `frame_id`
nor the `MsType::Precursor` the caller expects.

### 2. Reference-noise samplers do an inner sum that propagates `Unknown`

`rustdf/src/data/dia.rs:586-591`:
```rust
let mut sampled_frame = sampled_frames.remove(0);
for frame in sampled_frames {
    sampled_frame = sampled_frame + frame;
}
```
`TimsFrame::Add` (correctly) collapses `ms_type` to `MsType::Unknown`
when the two operands disagree, and keeps the LHS `frame_id` and
`retention_time`. So if the **first** sampled frame happens to be one
of the empty-fallback `Default`s (`frame_id=0, ms_type=Unknown`,
`retention_time=0.0`), the entire returned noise inherits
`frame_id=0` and `ms_type=Unknown` regardless of what the subsequent
picks contribute.

### 3. The outer `simulated + noise` then collapses the *real* frame's
   `ms_type`, and a leakage path lets the noise's `frame_id` / `RT`
   reach the writer

`packages/imspy-simulation/src/imspy_simulation/timsim/jobs/add_noise_from_real_data.py:118-120`:
```python
noise = acquisition_builder.tdf_writer.helper_handle.sample_precursor_signal(...)
r_list.append(frame + noise)
```
`frame + noise` should give `result.frame_id = frame.frame_id` (LHS),
but on the affected dataset all 115 corrupted output rows share
`Id=1, Time=0.0` — i.e. the noise's metadata is what landed in the
writer for those rows. Reading `add_real_data_noise_to_frames` and the
Rust `Add` impl alone does not explain how the LHS got overridden;
the most likely explanation is a separate code path (frame-builder
edge case for "no peptides in this frame's RT window" + noise mix
order) where the simulated `frame` itself reaches the loop already
in `Default` shape, after which `Default + noise` returns the noise's
metadata. Confirming this exhaustively requires running a small
instrumented midiA simulation; the fix in (1) and (2) below removes
the corruption regardless of which exact leak path the third
component takes.

## Patch (proposed)

The minimal high-leverage fix is at the sampler boundary. After this,
no leak path downstream can produce an output frame with mismatched
metadata, because the noise is guaranteed to carry the *requested*
type.

```rust
// rustdf/src/data/dia.rs   sample_precursor_signal
pub fn sample_precursor_signal(...) -> TimsFrame {
    let precursor_frames = self.meta_data.iter().filter(|x| x.ms_ms_type == 0);

    // Re-roll on empty: empty raw frames give us a Default-shaped
    // TimsFrame and corrupt the inner accumulator. Keep drawing until
    // we have `num_frames` non-empty samples (bounded retry to avoid
    // infinite loops if the reference's MS1 pool is mostly empty).
    let mut rng = rand::thread_rng();
    let mut sampled_frames: Vec<TimsFrame> = Vec::with_capacity(num_frames);
    let mut tries = 0usize;
    let mut iter_pool: Vec<&FrameMeta> = precursor_frames.collect();
    while sampled_frames.len() < num_frames && tries < num_frames * 8 {
        let frame = iter_pool.choose(&mut rng).unwrap();
        let candidate = self
            .loader
            .get_frame(frame.id as u32)
            .filter_ranged(0.0, 2000.0, 0, 1000, 0.0, 5.0, 1.0, max_intensity, 0, i32::MAX)
            .generate_random_sample(take_probability);
        tries += 1;
        if candidate.ms_type == MsType::Unknown || candidate.scan.is_empty() {
            continue;  // empty-fallback or otherwise corrupted; skip
        }
        sampled_frames.push(candidate);
    }
    if sampled_frames.is_empty() {
        // Defensive: return a *typed* empty frame so the caller's Add
        // can't collapse ms_type later.
        return TimsFrame::new(0, MsType::Precursor, 0.0,
                              vec![], vec![], vec![], vec![], vec![]);
    }

    let mut sampled_frame = sampled_frames.remove(0);
    for frame in sampled_frames {
        sampled_frame = sampled_frame + frame;
    }
    // Belt-and-braces: stamp the type explicitly, in case the inner
    // accumulator somehow collapsed it.
    sampled_frame.ms_type = MsType::Precursor;
    sampled_frame
}
```

Apply the analogous change to `sample_fragment_signal` with
`MsType::FragmentDia` as the stamp.

### Why these specific changes

- **Skip empty / Unknown-typed candidates** before they enter the
  inner accumulator. Closes the root leak from `get_frame()`'s
  Default-fallback.
- **Bounded retry** rather than unbounded `while` — if the reference's
  MS1 pool is genuinely degenerate (e.g. a blank gradient with very
  few non-empty MS1 frames), the sampler degrades gracefully instead
  of hanging.
- **Belt-and-braces stamp** after the accumulation — purely defensive
  in case a future change inside the loop introduces a new Unknown
  pathway. Trivially cheap; documents the post-condition.
- **Defensive empty-frame return** when the pool yielded zero usable
  samples. Returns a *typed* empty frame so the caller's `frame + noise`
  doesn't collapse the simulated frame's `ms_type`.

### Why *not* fix `TimsFrame::Add`

The Add semantics are correct as-is (David's point — `precursor +
fragment` legitimately has no defined `ms_type`, `Unknown` is the
right answer). The bug is that the operand pairs reaching Add are
ones that should never have been mixed in the first place.

### Why *not* fix `get_frame()`'s empty-fallback

`get_frame()` is a general-purpose reader; returning a Default-shaped
frame is a reasonable signal for "no data at this offset" that callers
*should* be checking. Demanding callers check `ms_type != Unknown`
before consuming the frame is more honest than papering over the
empty case at the reader level.

## Test

Add a unit test under `rustdf/tests/sim_noise_sampler.rs`:

```rust
#[test]
fn sample_precursor_signal_never_emits_unknown_ms_type() {
    let handle = open_test_reference_with_some_empty_ms1_frames();
    for _ in 0..1000 {
        let noise = handle.sample_precursor_signal(5, 100.0, 0.1);
        assert_ne!(noise.ms_type, MsType::Unknown,
                   "sampler must stamp ms_type on returned noise");
    }
}
```

And the analogous test for `sample_fragment_signal`.

End-to-end regression: a small midiA sim run with `add_real_data_noise=True`
against a reference that has at least one empty MS1 frame must produce
an output `Frames` table with zero rows where `MsMsType=-1`.

## Open questions left for the owner

- Is there a non-noise codepath that could still produce `frame_id=1,
  Time=0.0, MsMsType=-1` on output frames? I narrowed the corruption to
  the noise sampler by elimination but the exact "noise frame_id leak
  into output" path I couldn't fully trace from static reading. The
  proposed patch should resolve it irrespective, since the sampler's
  return frame is now guaranteed to carry a defined `MsType`.
- Should `Frames.Id` be promoted to `PRIMARY KEY` in the writer's
  `to_sql` so this kind of corruption raises immediately on write?
  Loses some `to_sql` ergonomics but adds a strong invariant.
