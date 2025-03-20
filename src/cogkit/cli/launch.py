# -*- coding: utf-8 -*-


import click
from fastapi_cli.cli import _run

from cogkit import api
from cogkit.utils import resolve_path


@click.command()
@click.option("--host", default="127.0.0.1", help="The host to serve on, default: 127.0.0.1")
@click.option(
    "--port",
    type=click.IntRange(min=0, max=65_535),
    default=8000,
    help="The port to serve on, default: 8000",
)
@click.option(
    "--workers",
    type=click.IntRange(min=1),
    default=1,
    help="The number of worker process, default: 1",
)
def launch(
    host: str = "127.0.0.1",
    port: int = 8000,
    workers: int = 1,
) -> None:
    _run(
        path=resolve_path(api.__file__),
        host=host,
        port=port,
        reload=False,
        workers=workers,
        command="run",
    )
