__all__ = ["progress_bar"]

from contextlib import contextmanager

from dask.diagnostics.progress import ProgressBar


@contextmanager
def progress_bar(progress: bool = True):
    if progress:
        with ProgressBar():
            yield
    else:
        yield
