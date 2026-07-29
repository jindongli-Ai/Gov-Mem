from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class RuntimeStaticBoundaryTests(unittest.TestCase):
    def test_stateful_runtime_has_no_legacy_stage2_import(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(root / "scripts" / "check_stateful_policy_runtime.py")],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
