"""Exporter base class."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Exporter(ABC):
    """Abstract base class to handle the export of data.

    The `Exporter` class provides methods to save the transformed data to a
    file in a specified format (e.g., JSON or TXT).

    Attributes:
        export_format (ExportFormat): The format in which the data will be
        exported.
    """

    export_format: str

    @abstractmethod
    def write_data(
        self, data: Any, output_file: Path, name_of_output_property: str = ""
    ) -> dict:
        """Writes the provided data to an output file. This method must be
        implemented by subclasses to handle specific export formats.

        Arguments:
            data (Any): The data to be written to the file.
            output_file (Path): Path to the output file.
            name_of_output_property (str): Optionally the name of the output
                property.

        Returns:
            dict: A dictionary containing the exported data.
        """
        pass

    def _validate_outfile(self, output_file: Path) -> Path:
        """Validates the output file path and ensures the directory exists.
        This function checks if the output file has a suffix and appends the
        export format suffix if it does not. It also creates the output
        directory if it does not exist.

        Arguments:
            output_file (Path): The path to the output file.

        Returns:
            None
        """
        if not output_file.suffix:
            logging.warning(
                "Output file has no suffix. Appending the export format "
                "suffix."
            )
            output_file = output_file.with_suffix(f".{self.export_format}")
        if not output_file.parent.exists():
            logging.info(
                f"Creating directory {output_file.parent} for output file."
            )
            output_file.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Writing data to {output_file}")

        return output_file
