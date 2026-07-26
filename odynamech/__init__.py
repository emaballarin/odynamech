"""Open biomechanics data as a dynamical-systems and machine-learning testbed.

The OpenBiomechanics Project release is many bodies executing the same act with
markedly different parameters and initial conditions. That makes it a useful and
non-artificial testbed for models of dynamical evolution under inter-individual
variability: the decomposition into anatomical sites and discrete events is
native to the data rather than imposed on it, and the whole setup stays easy to
describe. This package exists for that purpose, not for sports analytics.

**No data ships here.** `odynamech` is MIT-licensed code that automates acquiring
the data from a source you nominate. The data itself is third-party, CC BY-NC-SA
4.0, non-commercial, and additionally excludes anyone affiliated with a
professional sports organisation or a financial-analysis firm. See
`odynamech.config.DATA_NOTICE`, or run `odynamech licence`.

Three acquisition paths, of which exactly one has a default URL:

    >>> import odynamech as odm
    >>> obp = odm.load()  # Driveline, the default
    >>> obp = odm.load(odm.RawMirror(base_url="https://…"))  # raw tables, your URL
    >>> obp = odm.load(odm.PackagedCorpus(url="https://…"))  # prebuilt corpus, your URL

    >>> p = obp["pitching"]
    >>> p.select(site="elbow").torch().shape  # doctest: +SKIP
    >>> train, val, test = p.split_by_athlete()  # athlete-disjoint by construction
"""

from .api import build
from .api import corpus_path
from .api import fetch
from .api import load
from .config import ATTRIBUTION
from .config import cache_root
from .config import DATA_NOTICE
from .config import raw_dir
from .config import tensors_dir
from .harmonise import harmonise
from .pack import pack
from .schema import build_schema
from .schema import channel_hash
from .schema import ChannelSpec
from .schema import UnclassifiedChannel
from .sources import DrivelineRelease
from .sources import PackagedCorpus
from .sources import RawMirror
from .sources import Source
from .store import ChannelHashMismatch
from .store import Store
from .torchio import collate
from .torchio import split_by_athlete
from .torchio import TrialDataset
from .verify import Report
from .verify import verify_corpus
from .view import EmptySelection
from .view import GestureView
from .view import HarmonisedView
from .view import OBP

__all__ = [
    "ATTRIBUTION",
    "DATA_NOTICE",
    "OBP",
    "ChannelHashMismatch",
    "ChannelSpec",
    "DrivelineRelease",
    "EmptySelection",
    "GestureView",
    "HarmonisedView",
    "PackagedCorpus",
    "RawMirror",
    "Report",
    "Source",
    "Store",
    "TrialDataset",
    "UnclassifiedChannel",
    "build",
    "build_schema",
    "cache_root",
    "channel_hash",
    "collate",
    "corpus_path",
    "fetch",
    "harmonise",
    "load",
    "pack",
    "raw_dir",
    "split_by_athlete",
    "tensors_dir",
    "verify_corpus",
]

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"
