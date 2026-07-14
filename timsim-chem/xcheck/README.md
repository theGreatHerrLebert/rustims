# xcheck — an independent oracle for the cleavage rules

The Monte Carlo test in `digest.rs` proves the **yield maths** is right. It says nothing
about whether we cut the protein in the right **places**. If our trypsin is subtly wrong,
every yield we compute is a correct number about the wrong peptides — and everything
downstream silently inherits it.

So this digests the same FASTA with **Sage** (independent, widely used, battle-tested) and
with us, and compares the peptide sets. This is the anti-self-consistency rule from
`TIMSIM_V2_PLAN.md` §6.1: expected values must come from an implementation other than the
one under test.

```bash
cargo run --release -- <fasta> [max_missed_cleavages]
```

## Result (human proteome + contaminants, 20,590 proteins)

| max missed cleavages | peptides | in Sage not ours | in ours not Sage |
|---|---|---|---|
| 0 | 556,494 | 0 | 0 |
| 1 | 1,499,745 | 0 | 0 |
| 2 | 2,516,825 | 0 | 0 |
| 3 | 3,443,877 | 0 | 0 |
| 5 | 4,821,538 | 0 | 0 |

Identical, every time.

## The one difference, and it is deliberate

At `max_missed = 2` we emit **2,529,220 occurrences** against 2,516,825 unique
(protein, peptide) pairs — **12,395 more**. Sage dedups within a protein
(`seen.insert(sequence)`, `enzyme.rs:316`), because a search engine does not care that a
peptide occurs twice in the same protein.

We do care: that is the occurrence-level contribution needed for protein-inference ground
truth. Same chemistry, different semantics — which is the whole thesis.
