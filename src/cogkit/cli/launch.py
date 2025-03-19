# -*- coding: utf-8 -*-


from pathlib import Path

import click
from dotenv import load_dotenv
from fastapi_cli.cli import _run

from cogkit import api
from cogkit.utils import resolve_path


@click.command()
@click.option(
    "--env_file", type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)
)
@click.option("--host", default="127.0.0.1", help="the host to serve on")
@click.option(
    "--port", type=click.IntRange(min=0, max=65_535), default=8000, help="the port to serve on"
)
@click.option(
    "--workers", type=click.IntRange(min=1), default=1, help="the number of worker process"
)
def launch(
    env_file: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    workers: int = 1,
) -> None:
    if env_file is not None:
        load_dotenv(resolve_path(env_file))

    _run(
        path=resolve_path(api.__file__),
        host=host,
        port=port,
        reload=False,
        workers=workers,
        command="run",
    )
