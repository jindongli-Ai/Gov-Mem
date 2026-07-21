from __future__ import annotations

import importlib
import os
import sys

_module_name = __name__
_module_dir = os.path.abspath(os.path.dirname(__file__))
_repo_root = os.path.abspath(os.path.join(_module_dir, ".."))
_saved_path = list(sys.path)
try:
    sys.modules.pop(_module_name, None)
    sys.path = [
        path
        for path in sys.path
        if os.path.abspath(path or "") not in {
            _repo_root,
            os.path.join(_repo_root, "third_party", "python_deps", "gatemem_eval"),
        }
    ]
    _real_numpy = importlib.import_module(_module_name)
finally:
    sys.path = _saved_path

globals().update(_real_numpy.__dict__)
sys.modules[_module_name] = _real_numpy
