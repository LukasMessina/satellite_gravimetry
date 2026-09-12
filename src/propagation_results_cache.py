""" Module to cache propagation results to disk. """

from pathlib import Path
from typing import Any

import hashlib
import json
import time

import h5py
import numpy as np

# Files at or below this size are fingerprinted by content hash; larger files
# (e.g. SPICE kernels) are fingerprinted by size/mtime only,
# since hashing their full contents on every run would be prohibitively slow.
FILE_HASH_THRESHOLD_BYTES = 1_000_000


class PropagationResultsCache:
    """Caches orbit-propagation products, keyed by a hash of their inputs."""

    @staticmethod
    def fingerprint_file(file_path: Path) -> dict[str, Any]:
        """Identify a file by content hash, or by size/mtime if it is too large to hash cheaply."""

        file_path = Path(file_path)
        file_stat = file_path.stat()
        fingerprint: dict[str, Any] = {
            "path": str(file_path),
            "size_bytes": file_stat.st_size,
        }
        if file_stat.st_size <= FILE_HASH_THRESHOLD_BYTES:
            fingerprint["sha256"] = hashlib.sha256(file_path.read_bytes()).hexdigest()
        else:
            fingerprint["mtime_ns"] = file_stat.st_mtime_ns
        return fingerprint

    @staticmethod
    def compute_config_hash(propagation_config: dict[str, Any]) -> str:
        """Hash a JSON-serializable propagation-input config into a stable cache key."""

        canonical_json = json.dumps(propagation_config, sort_keys=True, default=str)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    @staticmethod
    def _cache_paths(cache_dir: Path, config_hash: str) -> tuple[Path, Path]:
        cache_dir = Path(cache_dir)
        return cache_dir / f"{config_hash}.h5", cache_dir / f"{config_hash}_config.json"

    @staticmethod
    def load(cache_dir: Path, config_hash: str) -> dict[str, Any] | None:
        """Load cached propagation products for config_hash, or None if nothing is cached."""

        h5_path, _ = PropagationResultsCache._cache_paths(cache_dir, config_hash)
        if not h5_path.exists():
            return None

        with h5py.File(h5_path, "r") as h5_file:
            if h5_file.attrs["config_hash"] != config_hash:
                raise ValueError(
                    f"Cache file {h5_path} contains a config_hash that does not match "
                    f"the requested key ({config_hash}); the cache directory may be corrupted."
                )

            guidance_log_json = h5_file["guidance_log_json"][()]
            if isinstance(guidance_log_json, bytes):
                guidance_log_json = guidance_log_json.decode("utf-8")

            return {
                "states_array": h5_file["states_array"][()],
                "dependent_variables_array": h5_file["dependent_variables_array"][()],
                "guidance_log": json.loads(guidance_log_json),
                "mean_grace_fo_orbital_period": float(h5_file.attrs["mean_grace_fo_orbital_period"]),
                "total_cpu_time": float(h5_file.attrs["total_cpu_time"]),
                "total_function_evaluations": int(h5_file.attrs["total_function_evaluations"]),
            }

    @staticmethod
    def save(
        cache_dir: Path,
        config_hash: str,
        propagation_config: dict[str, Any],
        states_array: np.ndarray,
        dependent_variables_array: np.ndarray,
        guidance_log: list[dict[str, float | bool]],
        mean_grace_fo_orbital_period: float,
        total_cpu_time: float,
        total_function_evaluations: int,
    ) -> Path:
        """Persist propagation products to disk, keyed by config_hash."""

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        h5_path, json_path = PropagationResultsCache._cache_paths(cache_dir, config_hash)

        json_path.write_text(json.dumps(propagation_config, sort_keys=True, indent=2, default=str))

        with h5py.File(h5_path, "w") as h5_file:
            h5_file.create_dataset(
                "states_array",
                data=np.asarray(states_array, dtype=float),
                compression="gzip",
            )
            h5_file.create_dataset(
                "dependent_variables_array",
                data=np.asarray(dependent_variables_array, dtype=float),
                compression="gzip",
            )
            h5_file.create_dataset(
                "guidance_log_json",
                data=json.dumps(guidance_log),
                dtype=h5py.string_dtype(),
            )
            h5_file.attrs["config_hash"] = config_hash
            h5_file.attrs["mean_grace_fo_orbital_period"] = float(mean_grace_fo_orbital_period)
            h5_file.attrs["total_cpu_time"] = float(total_cpu_time)
            h5_file.attrs["total_function_evaluations"] = int(total_function_evaluations)
            h5_file.attrs["created_at_unix"] = time.time()

        return h5_path
