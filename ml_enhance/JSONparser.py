import json
from pathlib import Path
from typing import Any

import numpy as np


class NumpyJSONCache:
    """Tiny helper for caching NumPy arrays in human-readable JSON files.

    Example:
    --------
    >>> cache = NumpyJSONCache("cache/big_array.json")
    >>> if not cache.exists():
    >>>     cache.dump(my_array)   # save
    >>> array_again = cache.load()  # retrieve later
    """

    def __init__(self, filepath: Path | str) -> None:
        self.filepath = Path(filepath) if isinstance(filepath, str) else filepath

    # --------------------------------------------------------------------- #
    # public API                                                            #
    # --------------------------------------------------------------------- #
    def dump(self, obj: dict, *, indent: int | None = None) -> None:
        """Serialise *array* to disk in JSON format."""
        payload = (
            {key: self._pack(val) if isinstance(val, np.ndarray) else val for key, val in obj.items()}
            if isinstance(obj, dict)
            else self._pack(obj)
        )

        self._ensure_parent_dir()
        self._write_json(payload, indent=indent)

    def load(self) -> np.ndarray:
        """Load the array from JSON, raising FileNotFoundError if missing."""
        payload = self._read_json()
        if self._looks_like_dict(payload):
            return {key: self._unpack(val) for key, val in payload.items()}
        return self._unpack(payload)

    def exists(self) -> bool:
        """Return True if the cache file is already on disk."""
        return self.filepath.is_file()

    # --------------------------------------------------------------------- #
    # internal helpers                                                      #
    # --------------------------------------------------------------------- #
    def _pack(self, arr: np.ndarray) -> dict:
        """Build a JSON-serialisable dict from *arr*.

        For complex arrays we store real and imag parts separately.
        """
        payload: dict[str, Any] = {
            "dtype": str(arr.dtype),
            "shape": arr.shape,
        }

        if arr.dtype.kind == "c":  # complex
            payload["is_complex"] = True
            payload["real"] = arr.real.ravel().tolist()
            payload["imag"] = arr.imag.ravel().tolist()
        else:  # real/int/bool/etc.
            payload["is_complex"] = False
            payload["data"] = arr.ravel().tolist()

        return payload

    def _unpack(self, payload: dict) -> np.ndarray:
        """Reconstruct the NumPy array from a payload dict."""
        dtype = np.dtype(payload["dtype"])
        shape = tuple(payload["shape"])

        if payload.get("is_complex"):  # complex case
            real = np.array(payload["real"], dtype=dtype).reshape(shape)
            imag = np.array(payload["imag"], dtype=dtype).reshape(shape)
            return real + 1j * imag
        # real case
        return np.array(payload["data"], dtype=dtype).reshape(shape)

    def _looks_like_dict(self, payload: dict) -> bool:
        """Check if payload is a dict-of-dicts (multiple arrays)."""
        return isinstance(payload, dict) and all(isinstance(v, dict) and "dtype" in v for v in payload.values())

    def _ensure_parent_dir(self) -> None:
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def _write_json(self, payload: dict, *, indent: int) -> None:
        with self.filepath.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent)

    def _read_json(self) -> dict:
        with self.filepath.open("r", encoding="utf-8") as f:
            return json.load(f)
