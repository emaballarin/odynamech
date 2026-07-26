"""PyTorch interop: per-trial `Dataset`, default collation, and athlete-disjoint splits."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from .view import GestureView


class TrialDataset(Dataset):
    """One item per trial, materialised once from the view it wraps."""

    def __init__(self, view: GestureView, *, return_mask: bool = True, targets: Sequence[str] = ()) -> None:
        """Materialise `view` and cache the tensors the dataset serves."""
        self.view = view
        self.return_mask = return_mask
        self.targets = tuple(targets)
        self.x = view.torch()
        self.valid = view.mask() if return_mask else None
        self.ev = view.events()
        frame = view.trials
        self.trial_id = frame["trial_id"].astype("str").tolist()
        self.athlete_id = frame["athlete_id"].astype("str").tolist()
        missing = [t for t in self.targets if t not in frame.columns]
        if missing:
            raise KeyError(f"targets not present in the trial index: {missing}")
        self.target_values = {t: torch.as_tensor(frame[t].to_numpy(dtype=np.float32)) for t in self.targets}

        # `.torch()` under nans('drop') can remove rows; the label columns must follow.
        if self.x.shape[0] != len(self.trial_id):
            raise ValueError(
                f"materialised {self.x.shape[0]} trials but the index holds {len(self.trial_id)}. "
                "Resolve the NaN policy with .complete() or .nans('drop') applied before .dataset()."
            )

    def __len__(self) -> int:
        """Number of trials."""
        return self.x.shape[0]

    def __getitem__(self, i: int) -> dict[str, Any]:
        """One trial: `(T, C)` signal plus its mask, ids, events and targets."""
        item: dict[str, Any] = {
            "x": self.x[i],
            "trial_id": self.trial_id[i],
            "athlete_id": self.athlete_id[i],
            "events": self.ev[i],
        }
        if self.return_mask and self.valid is not None:
            item["mask"] = self.valid[i]
        for name, values in self.target_values.items():
            item[name] = values[i]
        return item


def collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensor fields; leave string fields as lists.

    The aligned store is fixed-length, so no ragged collation is needed and no
    padding collator is provided.
    """
    out: dict[str, Any] = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        out[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else values
    return out


def make_dataloader(
    view: GestureView,
    batch_size: int = 32,
    *,
    return_mask: bool = True,
    targets: Sequence[str] = (),
    complete: bool = False,
    generator: torch.Generator | None = None,
    **kw,
) -> DataLoader:
    """A `DataLoader` over one view. Pass a `generator` explicitly when shuffling."""
    dataset = TrialDataset(view.complete() if complete else view, return_mask=return_mask, targets=targets)
    kw.setdefault("collate_fn", collate)
    if kw.get("shuffle") and generator is None:
        raise ValueError("shuffle=True requires an explicit generator; no fixed seed is fabricated here")
    if generator is not None:
        kw["generator"] = generator
    return DataLoader(dataset, batch_size=batch_size, **kw)


def split_by_athlete(
    view: GestureView, fractions: tuple[float, ...] = (0.8, 0.1, 0.1), seed: int = 0
) -> tuple[GestureView, ...]:
    """Split athlete-disjointly, so multiple trials per athlete cannot leak across.

    A random *trial* split would be optimistic: most athletes contribute several
    trials. The athlete sets are asserted pairwise disjoint before returning.
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {fractions} summing to {sum(fractions)}")

    frame = view.trials
    athletes = np.array(sorted(frame["athlete_id"].astype("str").unique()))
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    rng.shuffle(athletes)

    # Cut on cumulative fractions rather than rounding each share independently,
    # so the boundaries cannot drift apart from the requested proportions.
    cuts = np.round(np.cumsum(fractions[:-1]) * len(athletes)).astype(np.int64)
    groups = np.split(athletes, cuts)

    views = []
    for group in groups:
        keep = frame["athlete_id"].astype("str").isin(set(group.tolist())).to_numpy()
        views.append(replace(view, _rows=view._rows[keep]))

    seen = [set(v.trials["athlete_id"].astype("str")) for v in views]
    for i in range(len(seen)):
        for j in range(i + 1, len(seen)):
            overlap = seen[i] & seen[j]
            if overlap:
                raise AssertionError(f"athlete leakage between splits {i} and {j}: {sorted(overlap)[:5]}")
    return tuple(views)
