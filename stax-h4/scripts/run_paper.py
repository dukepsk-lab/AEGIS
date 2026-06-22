"""P1/P0 entry point: run the bar-close event loop in paper mode against
the live IUX MT5 terminal (no real orders — decisions are logged only
until P3 execution wiring lands)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stax.engine import run

if __name__ == "__main__":
    run()
