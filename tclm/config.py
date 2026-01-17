from pathlib import Path
from dataclasses import dataclass


@dataclass
class Paths:
    root: Path = Path(__file__).resolve().parent.parent
    data: Path = root / "data"
    artifacts: Path = root / "artifacts"
    outputs: Path = root / "outputs"
    plots: Path = outputs / "plots"


def ensure_dirs() -> None:
    for p in [Paths.data, Paths.artifacts, Paths.outputs, Paths.plots]:
        p.mkdir(parents=True, exist_ok=True)


