"""Self-owned configured entry; the original validation script stays unchanged."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reconlab.entry import main


if __name__ == "__main__":
    raise SystemExit(main(project_root=ROOT))
