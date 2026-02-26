"""Module containing general functions that are used throughout the project."""

from _collections_abc import Callable, Iterable
from typing import Any

from joblib import Parallel, delayed
from tqdm import tqdm


def parallelize(
    func: Callable[[Any], Any], iterable: Iterable[Any], n_jobs: int = 4, backend: str = "loky"
) -> list[Any]:
    """Parallelize a function for a given array.

    Args:
        func (Callable[[Any], Any]): The function that needs to be applied in parallel.
        iterable (Iterable[Any]): The array to which the function needs to be applied in parallel.
        n_jobs (int): The number of jobs (cpus) that are used to perform the task in parallel.
        backend (str): The backend that is used by joblib to perform the paralellization.

    Returns:
        list[Any]: A list of results from the function applied on the iterable
    """
    return Parallel(n_jobs=n_jobs, backend=backend)(delayed(func)(item) for item in tqdm(iterable))
