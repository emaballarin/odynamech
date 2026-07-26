# `odynamech` — project notes

Code-only Python package that acquires the OpenBiomechanics Project (OBP) release
and turns it into a tensor corpus for **dynamical-systems and machine-learning
research**. See `README.md` for user-facing documentation; this file records
structure, conventions and the domain facts a contributor needs.

## Framing

The data is a testbed, not a subject. It was chosen because it is the dynamical
evolution of many bodies with large inter-individual variability, is explainable
to a layman, and decomposes naturally into points of interest and parts. It is
**not** a sports-analytics package and should never be described as one.

## Structure

```
odynamech/
  config.py      paths, upstream coordinates, licence and attribution text
  sources.py     the three acquisition paths, as frozen dataclasses
  fetch.py       stdlib download (resume, retry, sha256) + safe extraction
  api.py         load / fetch / build — the short path most users need
  schema.py      channel parsing, site vocabulary, canonical ordering, hashing
  build.py       ragged + aligned + full-rate force-plate construction
  pack.py        single-file pack, and its round-trip verification
  store.py       safetensors I/O, dir-or-pack opening, hash check
  harmonise.py   semantic-tuple intersection with back-pointers
  view.py        OBP, GestureView, HarmonisedView
  torchio.py     Dataset, collate, athlete-disjoint splits
  verify.py      the structural gates, as a library
  cli.py         `odynamech` console entry point
tests/
  test_sources.py  URL-default policy               (data-free)
  test_fetch.py    download/extract machinery       (data-free, file:// fixtures)
  test_schema.py   channel parsing, byte round-trip (data-free)
  test_corpus.py   the access layer                 (needs ODYNAMECH_TEST_CORPUS)
```

Conventions follow `jrmt` and `scfreg`: hatchling + hatch-vcs with the version
file at `odynamech/_version.py`, flat package directory, one import per line in
`__init__.py`, ruff via `~/ruffconfigs/ebdefault/ruff.toml`, GemFury push from
`.github/workflows/build.yml`.

## The URL-default rule

This is the package's central promise, and the tests enforce it:

- `DrivelineRelease` — **the only source with a default URL.**
- `RawMirror(base_url=...)` — required, no default, empty string rejected.
- `PackagedCorpus(url=...)` — required, no default, empty string rejected.

A mirror is never contacted unless it was named. Do not add a default, an
environment-variable fallback, or a "helpful" guess to the latter two.

## Testing

The suite is **data-free by default** and CI never downloads anything: the data
is third-party and non-commercially licensed. Run the access-layer tests locally
against a built corpus:

```bash
ODYNAMECH_TEST_CORPUS=~/.cache/odynamech/tensors/odynamech_corpus.safetensors \
ODYNAMECH_TEST_RAW=~/.cache/odynamech/raw \
    pytest
```

`ODYNAMECH_TEST_RAW` is optional; it enables the source-level NaN-structure gate.

## Domain facts, measured from `dataset-v1` on 24/07/2026

Probed directly, not taken from upstream documentation.

|                   | pitching                  | hitting                     |
| ----------------- | ------------------------- | --------------------------- |
| trials / athletes | 411 / 100                 | 677 / 98                    |
| marker rate       | 360 Hz                    | **360 Hz** (not transposed) |
| aligned shape     | `(411, 398, 286)`         | `(677, 409, 162)`           |
| anchor event      | `BR_time`                 | `contact_time`              |
| force-plate ratio | 3× for all 403 GRF trials | **1× and 3×, per trial**    |
| padded fraction   | 0.39 %                    | 0.48 %                      |
| dead channels     | 7                         | 7                           |

Harmonised common set: **66** channels (27 angle + 21 velo + 12 landmark + 6 GRF).
Single-file pack: ~1018 MB.

### Upstream claims that are false

1. **"No NaNs in any measurement channel."** True for `joint_angles` only.
   `joint_velos` is NaN at frames `0, n-1`; `forces_moments` at `0, 1, n-2, n-1`
   (differentiation edges, structurally undefined rather than missing);
   `energy_flow`'s joint-energy channels across whole GRF-missing trials, since
   they are force-derived. Hitting `landmarks` has genuine interior marker
   dropouts in 9 trials. `verify.py` asserts this _structure_, which is strictly
   stronger than asserting zero: corruption still fails, physics does not.
2. **A single force-plate ratio per gesture.** Hitting mixes 1× and 3×, so
   `fp_ratio` is an `(N,)` int32 tensor and `0` means no force plate.
3. **`landmarks` has 60 channels.** It has 54 — 18 points × 3 axes.
4. **`(group, site, axis, ref_segment)` identifies a channel.** It does not:
   `elbow_energy_transfer_stp` / `_jfp` / `_generated` collide, as do
   `thorax_ap` / `_dist` / `_prox`. Hence the `subtype` column, making the tuple
   `(group, site, subtype, axis, ref_segment)`.

### Gotchas worth not rediscovering

- **Never infer the clock from `diff(time)`.** `time` is written to 4 decimals,
  so differences alternate `0.0027`/`0.0028` at 360 Hz. Determine the rate by
  requiring `np.rint(time * rate)` to be exactly `0..len-1` within every trial.
  That sweep is also what catches a mixed rate — it returns the empty set rather
  than a plausible wrong answer.
- **Do not rank trials with `pd.factorize`.** It orders by first appearance, not
  canonically, which silently misaligns the CSR offsets against the row blocks.
  Rank from the sorted trial-id list. This passed on pitching purely because that
  CSV happened to be sorted, and failed loudly on hitting.
- **`safetensors` partial reads only help on the leading axis.** A contiguous
  trial-axis slice costs ~2 MiB on a 154 MB tensor; a thin stripe along the
  channel axis touches every page and costs the same as a full read.
- **Each gesture must unzip into its own directory.** Upstream reuses table names
  across gestures, so flattening lets `hitting/joint_angles.csv` overwrite
  `pitching/joint_angles.csv`.

## Deliberate non-goals

Raw C3D ingestion (redundant with `landmarks`, ~0.6 GB), markerless/CV material,
the `high_performance` scalar module, and any modelling. Cross-gesture
harmonisation is in scope but strictly channel-axis, intersection-based and
reversible; harmonising the time axis, or a union with imputation, is not.

Guessing the arm correspondence (throwing/glove ↔ lead/rear) or the landmark side
correspondence (left/right ↔ lead/rear) is excluded not because it is hard but
because it is a handedness-and-role question the column names do not answer. The
mapping tables in `harmonise.py` are deliberately empty and default to excluded.
