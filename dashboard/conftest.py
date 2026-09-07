# Redundant for `python -m pytest` runs from the repo root -- the root
# pytest.ini's `pythonpath = dashboard` already covers that. Kept so
# `resultsboard` is still importable for editors/tools (IDE test runners,
# linters, etc.) that invoke pytest from within dashboard/ or otherwise
# don't read pytest.ini.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
