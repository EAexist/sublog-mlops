# Typer CLI — thin wrapper for local dev, calls same functions as DAG

import logging

import typer

app = typer.Typer()
logging.basicConfig(level=logging.INFO)


@app.command()
def run() -> None:
    """Run full pipeline locally (no Docker/Airflow)."""
    # TODO: generate_dataset → run_benchmarks → compute_metrics → score_and_rank → generate_report
    typer.echo("Pipeline not yet wired; implement using llm_benchmark.*")


if __name__ == "__main__":
    app()
