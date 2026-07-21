from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
SRC_PACKAGE_ROOT = PROJECT_ROOT / "src" / "gov_mem"
LOCAL_DEPS = PROJECT_ROOT / "third_party" / "python_deps" / "gatemem_eval"

if LOCAL_DEPS.exists() and str(LOCAL_DEPS) not in sys.path:
    sys.path.insert(0, str(LOCAL_DEPS))

if SRC_PACKAGE_ROOT.exists():
    __path__.append(str(SRC_PACKAGE_ROOT))
