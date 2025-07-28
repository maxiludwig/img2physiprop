"""TXT Exporter for exporting data to text files."""

from pathlib import Path
from typing import Any

from i2pp.core.exporters.exporter import Exporter


class TxtExporter(Exporter):
    """Exporter for writing data to text files."""

    export_format = "txt"

    def write_data(
        self, data: Any, output_file: Path, name_of_output_property: str = ""
    ) -> dict:
        """Writes the provided data to an output text file.

        Arguments:
            data (Any): The data to be written to the file. For TXT format
                export, it needs to be a string.
            output_file (Path): Path to the output file.
            name_of_output_property (str): The name of the output property.
                Not used for TXT format export.

        Returns:
            dict: A dictionary containing the exported data. For TXT format,
                it returns an empty dictionary.
        """
        output_file = self._validate_outfile(output_file)

        err_msg = (
            "You specified a TXT export format. In this case, the user "
            "function must return a string."
        )
        assert isinstance(data, str), err_msg
        with open(output_file, "w") as txt_file:
            txt_file.write(data)
        return {}
