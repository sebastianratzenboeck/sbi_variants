"""Train the original SimFormer-style conditional model."""

try:
    from .train_mock_galaxy import main
except ImportError:
    from train_mock_galaxy import main


if __name__ == "__main__":
    main()
