"""Train NF posterior p(theta|x_obs) with the train_nf_zuko_theta preset."""

import sys
from pathlib import Path

try:
    from .train_sbi_posterior import main as train_main
except ImportError:
    from train_sbi_posterior import main as train_main


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--config" not in args:
        default_cfg = Path(__file__).resolve().parent / "configs" / "train_nf_zuko_theta.json"
        args = ["--config", str(default_cfg), *args]

    if "--method" not in args:
        args = ["--method", "normalizing_flow", *args]

    sys.argv = [sys.argv[0], *args]
    train_main()
