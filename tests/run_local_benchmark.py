import asyncio
import logging
from pathlib import Path

from dags.benchmark_dag import step_run_benchmarks


def main():
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", force=True
    )

    for name in logging.Logger.manager.loggerDict:
        logging.getLogger(name).setLevel(logging.DEBUG)

    langfuse_logger = logging.getLogger("langfuse")
    langfuse_logger.setLevel(logging.DEBUG)

    asyncio.run(step_run_benchmarks(run_dir=Path("data"), use_dataset_cache=True))


if __name__ == "__main__":
    main()
