import argparse
import sys
from pathlib import Path

# Add the src directory to the path to allow direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src_cascades.runner import run_experiment_from_config

def main():
    parser = argparse.ArgumentParser(description="Run an SRC experiment from a configuration file.")
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    config_path = Path(args.config_file)
    if not config_path.is_file():
        print(f"Error: Configuration file not found at '{config_path}'")
        sys.exit(1)

    print(f"Loading experiment from: {config_path}")
    run_experiment_from_config(str(config_path))
    print("Experiment finished.")

if __name__ == "__main__":
    main()