# Streamlit dashboard (optional)

import logging

logger = logging.getLogger(__name__)


def run_dashboard(results_path: str | None = None) -> None:
    """Launch Streamlit dashboard. Optional entrypoint."""
    # TODO: streamlit run ...
    logger.info("Dashboard not yet implemented; results_path=%s", results_path)
