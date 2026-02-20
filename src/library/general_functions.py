from joblib import Parallel, delayed
from _collections_abc import Callable, Iterable
from typing import Any

def parallelize(func: Callable[[Any], Any], iterable: Iterable[Any], n_jobs=4, backend="loky") -> list[Any]:
    return Parallel(n_jobs=n_jobs, backend=backend)(delayed(func)(item) for item in iterable)