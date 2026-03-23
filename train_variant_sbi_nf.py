"""Train direct SBI posterior p(theta|x_obs) with normalizing-flow output head."""

import sys

try:
    from .train_sbi_posterior import main as train_main
except ImportError:
    from train_sbi_posterior import main as train_main


if __name__ == "__main__":
    if "--method" not in sys.argv:
        sys.argv = [sys.argv[0], "--method", "normalizing_flow", *sys.argv[1:]]
    train_main()
