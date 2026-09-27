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
    """Caches orbit-propagation products on disk, keyed by a hash of their inputs.

    Orbit propagation is deterministic given its inputs but expensive to (re)run.
    This class lets a caller check whether a propagation with a given set of
    inputs has already been run, load its saved results if so, or save fresh
    results for next time if not. See fingerprint_file/compute_config_hash for
    how the cache key is derived, and load/save for the cache entry itself.
    """

    @staticmethod
    def fingerprint_file(file_path: Path) -> dict[str, Any]:
        """Build an identifying fingerprint for a file, for use inside a cache key.

        Small files are fingerprinted by a SHA-256 hash of their full content, which
        detects any change reliably. Large files (e.g. multi-hundred-MB SPICE kernels)
        are fingerprinted by size and modification time instead, since hashing their
        full contents on every run would be prohibitively slow.

        Args:
            file_path (Path): Path to the file to fingerprint.

        Returns:
            dict[str, Any]: A dict with keys "path", "size_bytes", and either
                "sha256" (small files) or "mtime_ns" (large files).
        """

        file_path = Path(file_path)
        file_stat = file_path.stat()
        fingerprint: dict[str, Any] = {
            "path": str(file_path),
            "size_bytes": file_stat.st_size,
        }
        if file_stat.st_size <= FILE_HASH_THRESHOLD_BYTES:
            fingerprint["sha256"] = hashlib.sha256(file_path.read_bytes()).hexdigest()
        else:
            # Content hash would be too slow for very large files; fall back to a
            # cheap proxy that still changes whenever the file is rewritten.
            fingerprint["mtime_ns"] = file_stat.st_mtime_ns
        return fingerprint

    @staticmethod
    def compute_config_hash(propagation_config: dict[str, Any]) -> str:
        """Hash a JSON-serializable propagation-input config into a stable cache key.

        Args:
            propagation_config (dict[str, Any]): Nested dict describing every input
                that affects the orbit propagation (simulation window, spacecraft
                properties, force-model settings, guidance parameters, referenced
                file fingerprints, etc.). Must be JSON-serializable.

        Returns:
            str: A deterministic hex-encoded SHA-256 digest of the config. Identical
                configs (regardless of key insertion order) always produce the same
                hash, since keys are sorted before hashing.
        """

        # sort_keys=True guarantees the same config always serializes to the same
        # text, regardless of the order keys were inserted.
        canonical_json = json.dumps(propagation_config, sort_keys=True, default=str)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    @staticmethod
    def _cache_paths(cache_dir: Path, config_hash: str) -> tuple[Path, Path]:
        """Compute the on-disk paths for a given cache entry.

        Args:
            cache_dir (Path): Directory holding all cached propagation entries.
            config_hash (str): Hash identifying this particular cache entry.

        Returns:
            tuple[Path, Path]: The HDF5 data file path and the JSON config sidecar
                file path for this config_hash.
        """

        cache_dir = Path(cache_dir)
        return cache_dir / f"{config_hash}.h5", cache_dir / f"{config_hash}_config.json"

    @staticmethod
    def load(cache_dir: Path, config_hash: str) -> dict[str, Any] | None:
        """Load cached propagation products for config_hash, if they exist.

        Args:
            cache_dir (Path): Directory holding all cached propagation entries.
            config_hash (str): Hash identifying the cache entry to load, as produced
                by compute_config_hash.

        Returns:
            dict[str, Any] | None: A dict with keys "states_array",
                "dependent_variables_array", "guidance_log",
                "mean_grace_fo_orbital_period", "total_cpu_time", and
                "total_function_evaluations" if a matching cache entry exists on
                disk; otherwise None, signaling the propagation must be computed
                from scratch.

        Raises:
            ValueError: If a cache file exists at the expected path but the
                config_hash stored inside it does not match the requested key,
                indicating the cache directory may be corrupted.
        """

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
            # NOTE: h5py versions differ in whether a variable-length string dataset is
            # returned as str or bytes; normalize to str before decoding as JSON.
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
        """Persist propagation products to disk, keyed by config_hash.

        Writes two files: an HDF5 file holding the arrays/log/scalars needed to
        resume downstream processing without re-propagating, and a human-readable
        JSON sidecar holding the full input config, for inspection/debugging only
        (it is not read back by load()).

        Args:
            cache_dir (Path): Directory to write the cache entry into; created if
                it does not already exist.
            config_hash (str): Hash identifying this cache entry, as produced by
                compute_config_hash.
            propagation_config (dict[str, Any]): The full input config that produced
                these results, written to the JSON sidecar for reference.
            states_array (np.ndarray): Combined state history (time + Cartesian
                state per satellite) from the propagation.
            dependent_variables_array (np.ndarray): Combined dependent-variable
                history from the propagation.
            guidance_log (list[dict[str, float | bool]]): Record of every
                orbit-phasing maneuver planned/applied during propagation.
            mean_grace_fo_orbital_period (float): Mean orbital period across both
                GRACE-FO satellites, in seconds.
            total_cpu_time (float): Total propagation CPU time, in seconds.
            total_function_evaluations (int): Total number of dynamics function
                evaluations across the whole propagation.

        Returns:
            Path: Path to the HDF5 cache file that was written.
        """

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
            # guidance_log is a list of small per-maneuver dicts, not a numeric
            # array, so it is stored as a JSON text chunk rather than a dataset.
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
