"""
Main entry point for the bias correction pipeline.

This script initializes and runs the bias correction pipeline for climate model data.
"""

import sys
from pathlib import Path
import logging

# Add src to path BEFORE any local imports
SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

from pipeline import BiasCorrectPipeline  # noqa: E402


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """
    Main entry point for bias correction pipeline.

    Initializes the pipeline with configuration and executes the bias correction
    workflow. Handles errors gracefully and logs progress.

    Raises:
        FileNotFoundError: If config file is not found
        Exception: Any other errors during pipeline execution
    """
    config_path = "config.yaml"

    try:
        # Check if config file exists
        if not Path(config_path).exists():
            logger.error(f"Configuration file not found: {config_path}")
            logger.info("Please create a config.yaml file with the required settings")
            sys.exit(1)

        logger.info("Initializing bias correction pipeline...")
        pipeline = BiasCorrectPipeline(config_path=config_path)

        logger.info("Running bias correction...")
        pipeline.run()

        logger.info("Bias correction completed successfully")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration or data error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during pipeline execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
