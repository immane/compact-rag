"""CLI entry point for compact-rag using Typer."""

from __future__ import annotations

import typer

app = typer.Typer(help="compact-rag — Enterprise RAG system CLI")


@app.command()
def serve(
    config: str = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML config file",
    ),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Bind host"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes"),
):
    """Start the compact-rag API server."""
    from compact_rag.api.router import create_app
    from compact_rag.config.settings import get_settings

    settings = get_settings(config)
    fastapi_app = create_app(settings)

    import uvicorn

    uvicorn.run(
        fastapi_app,
        host=host,
        port=port,
        reload=reload,
        log_level=settings.log_level.lower(),
    )


@app.command()
def admin(
    config: str = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML config file",
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Admin bind host"),
    port: int = typer.Option(8501, "--port", "-p", help="Admin bind port"),
):
    """Start the Streamlit admin dashboard."""
    import subprocess
    import sys

    from compact_rag.config.settings import get_settings

    settings = get_settings(config)
    admin_path = "src/compact_rag/admin/app.py"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        admin_path,
        f"--server.port={port}",
        f"--server.address={host}",
    ]
    subprocess.run(cmd)


@app.command()
def version():
    """Show the compact-rag version."""
    from compact_rag import __version__

    typer.echo(f"compact-rag v{__version__}")


if __name__ == "__main__":
    app()
