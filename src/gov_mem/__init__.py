from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
LOCAL_DEPS_ROOT = PROJECT_ROOT / "third_party" / "python_deps"
if LOCAL_DEPS_ROOT.exists():
    for dep_dir in LOCAL_DEPS_ROOT.iterdir():
        if dep_dir.is_dir() and str(dep_dir) not in sys.path:
            sys.path.insert(0, str(dep_dir))
