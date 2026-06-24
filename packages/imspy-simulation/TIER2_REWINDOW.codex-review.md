The plan is directionally sound, but I would tighten the claims around 3a and be more skeptical about 3b. The highest-risk part is indeed redundant encoding of the isolation window, but the plan should separate “reaction metadata” from “scan acquisition/filter metadata” more explicitly.
**1. 3a in-place claim**
Yes, the central 3a claim looks correct if the window count is unchanged: the reaction record is fixed-size, and changing `EV_ISO_CENTER`, `EV_ISO_WIDTH`, and optionally collision energy is an in-place overwrite. No resize is needed for the reaction block itself.
But I would not phrase this as “only center/width change.” The reaction may be fixed-size, but the *semantic window* may be duplicated outside it. Fields that may co-vary:
- precursor/isolation center in the reaction record
- isolation width in the reaction record
- possibly activation/collision-energy fields if the scheme changes CE by window
- scan-event mass range block, if it contributes to filter or acquisition range
- scan-index low/high m/z, depending on whether those are observed peak bounds, acquisition bounds, or filter display bounds
- cached filter text, if present anywhere as serialized text rather than synthesized by RawFileReader
So: 3a is no-resize for scan events, but not necessarily “reaction-only.”
**2. Filter-string risk**
Yes, this is the right top risk. The validation must require both:
- `GetReaction(0)` reports the new precursor/isolation values.
- `GetFilterForScanNumber` reports the new `[lo-hi]`.
The exact fields to audit/update should be:
- Reaction record:
  - isolation center
  - isolation width
  - collision energy if intentionally changed
- Scan-event `nranges` / range block:
  - Determine whether this is scan acquisition mass range, isolation display range, or something else.
  - If a DIA MS2 event has a range equal to `[center - width/2, center + width/2]`, update it.
  - If it instead reflects fragment scan range, e.g. `150-2000`, do not tie it to isolation.
- Scan-index low/high m/z:
  - Be careful. In many formats, scan index low/high are observed data bounds, not isolation bounds.
  - If your existing repack path recomputes them from peaks, then for 3a they probably should remain peak-derived unless RawFileReader demonstrably uses them for filter `[lo-hi]`.
  - Do not blindly set scan-index low/high to the isolation window unless validation proves that is what Thermo expects.
- Cached filter string:
  - Search/parse for any serialized per-scan filter records. If present, update or invalidate them.
  - If RawFileReader synthesizes filters entirely from binary fields, there may be no cache. But the plan should explicitly verify this by byte-searching old window strings and checking relevant sections.
Other redundant locations worth checking:
- trailer extra fields such as isolation width, mono m/z, precursor m/z, or selected ion m/z
- status log only if it contains method/acquisition text, usually lower priority
- chromatogram/index structures if they group by precursor/window
- method/instrument method text, if downstream tools display the nominal DIA method
- DIA/window-group tables, if present in newer instruments/templates
The important distinction: `GetReaction` consistency is necessary; `GetFilterForScanNumber` consistency is not enough if chromatogram extraction or downstream indexing still sees old groups.
**3. 3a vs 3b value**
The split is technically sound. I would not skip 3a.
3a is valuable because it proves the redundancy map without introducing scan-count mutations. It gives you a constrained validation surface: same scans, same packets, same cadence, same offsets except patched fields. That is exactly the right place to learn which fields RawFileReader uses for filter synthesis.
But the plan should be honest that 3a does not unlock “arbitrary DIA window schemes.” It unlocks same-cardinality retargeting: shift, widen, narrow, overlap changes, or map an existing N-window cycle onto another N-window scheme. True 8 Th to 4 Th over the same range is 3b.
I would keep 3a as a deliberate proving step, then do 3b only after field redundancy is fully characterized.
**4. 3b hidden hazards**
`repack_many` helps with data-section relocation, but 3b is more than “resize events and packets.” Hazards I would call out:
- Scan numbering: Thermo APIs often assume dense scan numbers with consistent offsets across event, index, params, trailer, status, and data.
- Retention time / cycle structure: duplicated scans need plausible RTs, scan times, injection times, and TIC/base peak metadata.
- GenericRecord scan-params stream: this may not be a simple fixed array. It may contain per-scan variable records whose offsets/counts are mirrored elsewhere.
- Trailer extra: precursor fields, isolation width, injection time, charge state, master scan number, dependent scan info may need cloning/patching.
- Scan-event segment/dependent-scan fields: copied DDA structures can make DIA look like dependent MS/MS unless cleaned.
- MS1/MS2 cadence: inserting/removing MS2 scans changes cycle time assumptions, scan numbers between MS1s, and possibly controller event numbering.
- Chromatogram indexes: any precomputed BPC/TIC/SRM/precursor chromatograms may become inconsistent.
- DIA/window-group tables: if the file has explicit SWATH/DIA grouping metadata, this is likely the hardest hidden dependency.
- Checksums/section sizes: not just global Adler; any per-section or per-record length tables must be updated.
- Packet content: a newly inserted MS2 scan needs valid profile/centroid payload, trailer, params, index stats, and timestamps. Zero packets are risky unless already proven readable.
For 3b, copying a real same-MS-level skeleton is the right instinct, but choose skeletons by cycle position/window group, not just MS level.
**5. Validation and DIA-vs-DDA suitability**
The simulator’s heuristic, “distinct centers >2000 and >10% of MS2,” sounds like a DDA rejection rule, not a validation rule for re-windowed DIA. A re-windowed DIA template should still pass if the rewritten MS2 scans have a small repeated set of centers. If validation is looking for “not DDA,” then it should assert something like:
- number of distinct MS2 centers is much smaller than number of MS2 scans
- centers repeat across cycles
- most MS2 scans have isolation width in the DIA-expected range
- cycle pattern is stable enough for the requested scheme
The plan’s validation should add:
- old window strings/values are absent where expected
- per-cycle window sequence equals the requested scheme
- every MS2 scan maps to exactly one declared window
- `GetReaction`, filter `[lo-hi]`, trailer precursor/isolation fields, and any decoded range blocks agree
- downstream simulator transmission changes as expected for peptides near old/new boundaries
**6. Sequencing**
I would revise the sequence:
1. Build a “window provenance” inspector first: for each MS2 scan, print reaction center/width, filter `[lo-hi]`, event ranges, scan-index low/high, trailer precursor/isolation fields, and any cached strings.
2. Run it on Fusion and Astral templates before mutation.
3. Implement 3a with only same-count schemes.
4. Validate old-vs-new boundary cases in RawFileReader and the simulator.
5. Only then design 3b, starting with duplicating/removing one MS2 scan in a tiny controlled cycle before full retiling.
Bottom line: keep 3a. It is not the final payoff, but it is the right diagnostic increment. The plan’s main correction is to downgrade “arbitrary DIA window schemes” to “same-cardinality re-windowing” for 3a, and to broaden the redundancy audit beyond filter strings into trailer, range blocks, scan params, and possible DIA grouping metadata.
