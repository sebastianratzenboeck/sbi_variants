"""Train direct SBI posterior p(theta|x_obs) with flow-matching output head."""

import sys

try:
    from .train_sbi_posterior import main as train_main
except ImportError:
    from train_sbi_posterior import main as train_main


if __name__ == "__main__":
    if "--method" not in sys.argv:
        sys.argv = [sys.argv[0], "--method", "flow_matching", *sys.argv[1:]]
    train_main()
