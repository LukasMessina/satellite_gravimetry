from __future__ import annotations

import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DataDownloader:
    """
    High-level wrapper for the `podaac-data-downloader` CLI tool.

    This class provides a programmatic interface to query and download datasets
    from NASA Earthdata archives (via CMR) using the PO.DAAC bulk-data downloader.

    Parameters
    ----------
    collection : str
        CMR collection short name identifying the dataset to retrieve.

    output_directory : str | Path
        Local directory where downloaded files will be stored.

    start_date : str
        Start of the temporal query window, in ISO format.
        Example: "2019-01-01T00:00:00Z"

    end_date : str
        End of the temporal query window, in ISO format.
        Example: "2019-02-01T00:00:00Z"

    extensions : str, optional
        Regular expression used to filter files by extension.

    executable : str, optional
        Name or path of the downloader executable.
        Default: "podaac-data-downloader"

    granule_name : str, optional
        Pattern used to filter granules by name (granuleUR).

    provider : str, optional
        Data provider (DAAC) to query.

    limit : int, optional
        Maximum number of granules to download.

    verbose : bool, optional
        Enables verbose output from the downloader.
    """

    collection: str
    output_directory: str | Path
    start_date: str
    end_date: str
    extensions: str = ""
    executable: str = "podaac-data-downloader"
    granule_name: Optional[str] = None
    provider: Optional[str] = None
    limit: Optional[int] = None
    verbose: bool = False

    def build_command(self) -> list[str]:
        """Build the CLI command as a list of arguments."""
        cmd = [
            self.executable,
            "-c",
            self.collection,
            "-d",
            str(self.output_directory),
            "-sd",
            self.start_date,
            "-ed",
            self.end_date,
            "-e",
            self.extensions,
        ]

        if self.granule_name:
            cmd.extend(["-gr", self.granule_name])

        if self.provider:
            cmd.extend(["-p", self.provider])

        if self.limit is not None:
            cmd.extend(["--limit", str(self.limit)])

        if self.verbose:
            cmd.append("--verbose")

        return cmd

    def retrieve_l1b_data(self, capture_output: bool = False) -> subprocess.CompletedProcess:
        """
        Run the download command.

        Parameters
        ----------
        capture_output : bool
            If True, standard outputs and errors are captured and returned in the CompletedProcess.

        Returns
        -------
        subprocess.CompletedProcess
            The result of the subprocess execution.

        Raises
        ------
        RuntimeError
            If the command exits with non-zero status.
        """
        output_path = Path(self.output_directory)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            result = subprocess.run(
                self.build_command(),
                check=True,
                text=True,
                capture_output=capture_output,
            )
            return result
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Downloader executable not found: {self.executable!r}. "
                "Make sure the conda environment is active and "
                "`podaac-data-downloader` is installed."
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr_msg = exc.stderr.strip() if exc.stderr else "No stderr available."
            stdout_msg = exc.stdout.strip() if exc.stdout else "No stdout available."
            raise RuntimeError(
                "PO.DAAC download failed.\n"
                f"Exit code: {exc.returncode}\n"
                f"Standard Output:\n{stdout_msg}\n\n"
                f"Standard Error:\n{stderr_msg}"
            ) from exc

    def _list_downloaded_archives(self) -> list[Path]:
        """
        Return all downloaded archive in the output directory.

        This includes common compressed products such as .tgz, .tar.gz, .gz, and .zip.
        """
        output_directory_path = Path(self.output_directory)
        archive_files: list[Path] = []

        for path in output_directory_path.rglob("*"):
            if not path.is_file():
                continue

            file_name = path.name.lower()
            if (
                file_name.endswith(".tgz")
                or file_name.endswith(".tar.gz")
                or file_name.endswith(".gz")
                or file_name.endswith(".zip")
                or ".ascii." in file_name
            ):
                archive_files.append(path)

        return archive_files

    @staticmethod
    def _is_target_product(member_name: str, data_products: list[str]) -> bool:
        """Check whether an archive member should be extracted."""
        file_name = Path(member_name).name.upper()
        return any(data_product.upper() in file_name for data_product in data_products)
    
    @staticmethod
    def _extract_tar_member(
        tar: tarfile.TarFile,
        member: tarfile.TarInfo,
        destination: Path,
    ) -> None:
        """
        Safely extract one tar member into the destination directory.
        """
        member_name = Path(member.name).name
        if not member_name:
            return

        extracted_file = tar.extractfile(member)
        if extracted_file is None:
            return

        destination.mkdir(parents=True, exist_ok=True)
        output_file = destination / member_name

        with output_file.open("wb") as output:
            shutil.copyfileobj(extracted_file, output)

    def _retrieve_selected_data_products_from_archive(self, archive_path: Path, data_products: list[str]) -> None:
        """
        Extract only members related to requested data products from a .tgz/.tar.gz archive.

        After successful extraction, the archive is deleted.
        """
        suffixes_joined = "".join(archive_path.suffixes).lower()

        if not (suffixes_joined.endswith(".tgz") or suffixes_joined.endswith(".tar.gz")):
            raise ValueError(
                f"Unsupported archive format for extraction: {archive_path.name}. "
                "Expected .tgz or .tar.gz."
            )

        destination = archive_path.parent

        with tarfile.open(archive_path, "r:*") as tar_file:
            members_to_extract = [
                member
                for member in tar_file.getmembers()
                if member.isfile() and self._is_target_product(member.name, data_products)
            ]

            for member in members_to_extract:
                self._extract_tar_member(tar_file, member, destination)

        archive_path.unlink()

    def filter_l1b_data_archives(self, data_products: list[str]) -> None:
        """
        Extract data products, and delete the processed archives.
        """
        l1b_data_archives = self._list_downloaded_archives()

        for archive_path in l1b_data_archives:
            self._retrieve_selected_data_products_from_archive(archive_path, data_products)

    def get_l1b_data_products(self, data_products: list[str], capture_output: bool = False) -> subprocess.CompletedProcess:
        """
        Run the downloader and post-process the downloaded archives to retrieve the requested data products.

        Workflow
        --------
        - Download all matching archives
        - Extract only files specified by data_products and delete archives.

        Parameters
        ----------
        capture_output : bool
            If True, standard ouputs and errors are captured and returned in the CompletedProcess.

        Returns
        -------
        subprocess.CompletedProcess
            The result of the downloader execution.
        """
        result = self.retrieve_l1b_data(capture_output=capture_output)
        self.filter_l1b_data_archives(data_products=data_products)
        return result


if __name__ == "__main__":

    data_downloader = DataDownloader(
        collection="GRACEFO_L1B_ASCII_GRAV_JPL_RL04",
        output_directory="./data/grace_fo_l1b_ascii_grav_jpl_rl04_2019_01_01_2019_02_01",
        start_date="2019-01-01T00:00:00Z",
        end_date="2019-02-01T00:00:00Z",
        extensions="",
    )

    data_downloader.get_l1b_data_products(
        data_products = ["SCA1B", "GNI1B"]
    )