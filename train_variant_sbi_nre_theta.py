"""Train NRE/AMNRE ratio estimator with a default theta-focused config."""

import sys
from pathlib import Path

from train_sbi_nre import main as train_main


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--config" not in args:
        default_cfg = Path(__file__).resolve().parent / "configs" / "train_nre_balanced_theta.json"
        args = ["--config", str(default_cfg), *args]

    sys.argv = [sys.argv[0], *args]
    train_main()

