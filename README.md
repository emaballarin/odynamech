# ⚾ `odynamech`

Open biomechanics data for dynamical systems research (from Driveline Research & Development)

---

`odynamech` turns the [OpenBiomechanics Project](https://github.com/drivelineresearch/openbiomechanics)
release into a memory-mapped tensor corpus, and gives you structured, lazy access
to it with PyTorch interop.

## Why this data

Not for sports analytics. The OBP release is a large collection of **bodies
executing the same act with markedly different parameters and initial
conditions** — over a thousand trials, ~200 individuals, hundreds of synchronous
scalar signals sampled at 360 Hz. That makes it a useful testbed for
**dynamical-systems and machine-learning research**:

- **inter-individual variability is the point**, not noise to be averaged away —
  every athlete is a different realisation of the same underlying system;
- the decomposition into **points of interest and parts** (anatomical sites,
  discrete events like foot plant and ball release) is **native to the data**,
  not an abstraction imposed on it after the fact;
- the setup stays **easy to describe to a non-specialist**, which is rarer than
  it sounds for a benchmark with real dynamics in it.

The abstraction the package exposes follows that framing:

|             |                                                                                                                        |
| ----------- | ---------------------------------------------------------------------------------------------------------------------- |
| **gesture** | a global configuration — `pitching` or `hitting`                                                                       |
| **athlete** | an individual-level variation within a gesture: same act, different system                                             |
| **trial**   | one execution: `K` synchronous scalar signals over a common window                                                     |
| **channel** | one scalar signal, tagged by anatomical **site**, so several physical quantities at the same body part select together |

## Install

```bash
pip install odynamech
```

Requires Python ≥ 3.14. Dependencies: `numpy`, `pandas`, `pyarrow`,
`safetensors`, `torch`. The `odynamech` console script needs the `[cli]` extra.

## Licence — read this before fetching

**This package contains no data.** It is MIT-licensed code that automates
acquiring the data from a source _you_ nominate.

The **data** is a separate work, owned by Driveline Baseball, licensed
**CC BY-NC-SA 4.0** (Attribution, **NonCommercial**, **ShareAlike**), with an
additional exclusion beyond the plain CC terms: it may **not** be used by anyone
affiliated with a professional sports organisation or with a financial-analysis
firm. Confirm your own eligibility against the upstream terms — `odynamech`
cannot check this for you and does not try.

Two consequences that are easy to miss:

- **ShareAlike propagates.** A corpus built by `odynamech` is an _adaptation_. If
  you pass it on, it goes out under CC BY-NC-SA 4.0 and your recipients inherit
  both the rights and the obligations.
- **Attribution must state that changes were made.** `odynamech` makes material
  ones: it decimates the force plate to the marker clock, anchors and crops
  trials to a common window, edge-pads, and reorders the channel axis.

`odynamech licence` prints the full notice, which is also written to disk beside
any data it fetches, so a directory found later still says what it is.

## Getting the data

Three paths. **Exactly one has a default URL.**

```python
import odynamech as odm

# 1. The original Driveline release — the only source with a default URL.
obp = odm.load()

# 2. Raw upstream tables from somewhere else. No default; the URL is required.
obp = odm.load(odm.RawMirror(base_url="https://mirror.example.org/obp"))

# 3. An already-repackaged corpus. No default; the URL is required.
obp = odm.load(odm.PackagedCorpus(
    url="https://host.example.org/odynamech-corpus.tar.gz",
    sha256="9a8ace…",                        # optional but recommended
))
```

Paths 1 and 2 fetch ~0.9 GB of CSVs and build the corpus locally (a few minutes
of CPU). Path 3 skips the build entirely — the corpus arrives finished, so it is
the fast route and the one where `sha256` matters most.

Everything is cached under `$ODYNAMECH_HOME`, else `$XDG_CACHE_HOME/odynamech`,
else `~/.cache/odynamech`. A second `load()` opens the cache and touches nothing.
Pass `download=False` to forbid network access and fail instead.

Neither `RawMirror` nor `PackagedCorpus` falls back to upstream if handed an
empty URL — it raises. A mirror is never contacted unless it was named.

## Use

```python
p = obp["pitching"]
p.shape                    # (411, 398, 286) -> (trials, time, channels)
p.time                     # (398,) seconds, 0 at ball release
p.rate_hz                  # 360.0

# One measurement, one instant, one athlete
p.athlete("1031").value("1031_2", 0.0, "shoulder_angle_x")

# Everything about one body part — angle, velocity, force, moment, energy
p.select(site="elbow").channels[["name", "group", "subtype", "unit"]]

# Batchable tensor plus the mandatory validity mask
x = p.select(group=["angle"]).torch()      # (411, 398, 41) float32
m = p.select(group=["angle"]).mask()       # (411, 398) bool, False where padded

# Time axis
p.thin(4)                  # every 4th frame, anchor stays on grid
p.window(-0.4, 0.05)       # seconds relative to the anchor
p.frames(slice(0, 100))

# Escape hatches to the unprocessed record
p.ragged("1031_2")         # (550, 286) unaligned, unpadded, native length
p.fp_highrate("1031_2")    # (1650, 6) untouched 1080 Hz force plate

# Torch interop, athlete-disjoint by construction
train, val, test = p.split_by_athlete((0.8, 0.1, 0.1), seed=0)
loader = train.select(group=["angle"]).dataloader(batch_size=32, targets=["outcome"])
```

Views are **lazy and immutable**: every selector returns a new view holding only
index arrays, nothing is read until `.torch()` / `.numpy()` / iteration, and
selectors compose in any order.

### Across gestures

```python
h = obp.harmonised()                 # 66 channels common to both gestures
h["pitching"]                        # a native view, restricted — maps back losslessly
x, labels, frame = h.concat_gestures(window=(-0.5, 0.05))
```

Harmonisation is channel-axis only, by intersection on the semantic tuple
`(group, site, subtype, axis, ref_segment)`, and fully reversible. It never
touches the time axis — the gestures keep different anchors and different `T` —
so `concat_gestures` requires a common window and **asserts** matching `(T, C)`
rather than padding or warping.

### Missing values

Three modes, chosen at materialisation. Storage is always the faithful record.

```python
p.nans("keep")           # default — no edit
p.nans("interpolate")    # linear fill of gaps bracketed by known values
p.nans("drop")           # remove whole athletes or whole channels, cheaper wins
p.complete()             # dense, NaN-free, unpadded block; raises if empty
```

`interpolate` deliberately leaves leading and trailing gaps alone: a
finite-difference edge is _structurally undefined_, not missing, so there is
nothing to interpolate between. `drop` is the one selector whose effect is not
known until the values are read, so `.shape` reports the pre-drop shape — prefer
`.complete()` when the shape must be known up front.

### Handedness

Handedness is recorded but **no mirroring is applied by default**. Angle and
moment conventions are already handedness-adjusted upstream; `landmarks` are
global-frame positions and force-plate `y` is a lab-frame lateral axis, so
left-handers are geometrically mirrored in `y` for those groups only. Opt in with
`.torch(mirror_lefties=True)`, which negates `y` for the `landmark` and `grf`
groups and nothing else.

## Command line

```bash
odynamech info                                    # cache locations, what is present
odynamech licence                                 # the third-party data notice
odynamech fetch                                   # Driveline, the default source
odynamech fetch --raw-url https://…               # raw tables from elsewhere
odynamech load  --corpus-url https://… --sha256 … # prebuilt corpus
odynamech build                                   # build from already-fetched tables
odynamech verify                                  # structural gates; non-zero on failure
```

## What the data actually contains

Measured directly from `dataset-v1`, not taken from documentation. The points
most likely to surprise:

- **Measurement channels contain NaNs.** `joint_angles` has none in either
  gesture, but `joint_velos` is NaN at frames `0` and `n-1` of every trial
  (differentiation edge), `forces_moments` at `0, 1, n-2, n-1`, and
  `energy_flow`'s joint-energy channels across the whole of each trial that
  lacks force-plate data. Hitting `landmarks` additionally has genuine interior
  marker dropouts in 9 trials — worst is `440_6`, 147 consecutive frames.
- **The hitting force-plate rate is per trial**, not per gesture: 660 trials at
  3× (1080 Hz), 5 — all of one athlete — at 1× (360 Hz). `fp_ratio` is an `(N,)`
  tensor; `0` means no force plate.
- **Both gestures sample markers at 360 Hz.**
- **The two gestures name things differently.** Hitting `joint_velos` uses
  `_angular_velocity_` where pitching uses `_velo_`; hitting `landmarks` are
  anatomical (`rajc`, `lhjc`) where pitching's are role-based (`rear_ankle_jc`,
  `lead_hip`). The site parser resolves all of it and **raises on anything it
  cannot classify** rather than falling back to `unknown`.
- **The arms, and hitting's left/right landmarks, are excluded from the
  harmonised set by default.** Mapping throwing/glove ↔ lead/rear, or left/right
  ↔ lead/rear, is a handedness-and-role question, not a string question. The
  mapping tables live in `odynamech/harmonise.py` for a human to inspect.

Every store records `build_utc`, the upstream release tag, and a `sha256`
`channel_hash` over the ordered channel names; the loader recomputes and compares
on open, raising `ChannelHashMismatch` on drift.

## Verification

`odynamech.verify_corpus()` runs the structural gates — clock and join integrity,
channel-order drift, whether the alignment window amputates the event structure,
padding masquerading as signal, athlete leakage across splits, harmonisation
one-to-one-ness and round-trip identity, complete-cases emptying an axis, the
full-rate force-plate decimation identity, and the NaN structure against source.
It returns a report rather than raising, so one run surfaces every failure at
once.

## Licences

Code: MIT. Data: CC BY-NC-SA 4.0 plus the upstream exclusion — not distributed
here, and not relaxed by the code licence.
