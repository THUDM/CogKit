from pathlib import Path
import shutil
import torch.distributed as dist

from cogkit.finetune.logger import get_logger

from .dist import is_main_process


def check_path(
    path: str | Path | None,
    must_exists: bool = False,
    must_dir: bool = False,
    must_file: bool = False,
) -> None:
    if path is None:
        raise ValueError("Path is None")
    if isinstance(path, str):
        path = Path(path)
    if must_exists and not path.exists():
        raise FileNotFoundError(f"Path '{path}' does not exist.")
    if must_dir and not path.is_dir():
        raise FileNotFoundError(f"Path '{path}' is not a directory.")
    if must_file and not path.is_file():
        raise FileNotFoundError(f"Path '{path}' is not a file.")


def resolve_path(path: str | Path) -> str:
    if isinstance(path, str):
        path = Path(path)
    check_path(path)
    return str(path.expanduser().resolve())


def mkdir(path: str | Path) -> None:
    _logger = get_logger()
    if is_main_process():
        check_path(path)
        Path(resolve_path(path)).mkdir(parents=True, exist_ok=True)
        _logger.debug(f"Creating directory: {resolve_path(path)}")

    dist.barrier()


def touch(path: str | Path) -> None:
    _logger = get_logger()
    if is_main_process():
        check_path(path)
        Path(resolve_path(path)).touch()
        _logger.debug(f"Touching file: {resolve_path(path)}")

    dist.barrier()


def list_files(dir: str | Path | None, prefix: str = "checkpoint") -> list[str]:
    _logger = get_logger()
    if dir is None:
        _logger.warning("Directory is None, returning empty list")
        return []
    return [str(p) for p in Path(resolve_path(dir)).glob(f"{prefix}*")]


def rmdir(path: str | Path) -> None:
    _logger = get_logger()
    if is_main_process():
        check_path(path, must_exists=True, must_dir=True)
        Path(resolve_path(path)).rmdir()
        _logger.debug(f"Deleted empty directory: {resolve_path(path)}")

    dist.barrier()


def rmfile(path: str | Path, must_exists: bool = True) -> None:
    _logger = get_logger()
    if is_main_process():
        check_path(path, must_exists=must_exists, must_file=True)
        Path(resolve_path(path)).unlink()
        _logger.debug(f"Deleted file: {resolve_path(path)}")

    dist.barrier()


def rmtree(path: str | Path) -> None:
    """Recursively delete a directory tree."""
    _logger = get_logger()
    if is_main_process():
        path = Path(resolve_path(path))
        check_path(path, must_exists=True, must_dir=True)
        shutil.rmtree(path)
        _logger.debug(f"Recursively deleted directory: {path}")

    dist.barrier()


def delete_files(files: list[str], recursive: bool = True) -> None:
    for file in files:
        check_path(file, must_exists=True)
        path = Path(file)
        if path.is_dir():
            if recursive:
                rmtree(path)
            else:
                rmdir(path)
        else:
            rmfile(path)

    dist.barrier()
