"""JSON Exporter for exporting data to JSON files."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from i2pp.core.exporters.exporter import Exporter
from i2pp.core.utilities import make_json_serializable


class JsonExporter(Exporter):
    """Exporter for writing data to JSON files."""

    export_format = "json"

    def write_data(
        self, data: Any, output_file: Path, name_of_output_property: str = ""
    ) -> dict:
        """Writes the provided data to an output json file.

        Arguments:
            data (Any): The data to be written to the file. For JSON format
                export, it needs to be a numpy array. The first field must
                be integer-valued and named 'index'. The other fields can be
                of any type, but must be JSON serializable.
            output_file (Path): Path to the output file.
            name_of_output_property (str): The name of the output property.
                This is used as the key in the JSON output.

        Returns:
            dict: A dictionary containing the exported data. For JSON
                format, it has the output property name as the key and the
                processed data as the value.
        """
        self._validate_outfile(output_file)

        assert isinstance(data, np.ndarray), (
            "You specified a JSON export format. In this case, the user "
            "function must return a structured numpy array. First field "
            "must be integer-valued and named 'index'. The other fields "
            "can be of any type, but must be JSON serializable."
        )
        assert data.dtype.names is not None, (
            "The structured numpy array must have named fields. "
            "Adapt the user function."
        )
        assert np.issubdtype(data.dtype[0], np.integer), (
            "The first field of the structured numpy array must be "
            "integer-valued. Adapt the user function."
        )
        assert data.dtype.names[0] == "index", (
            "The first field of the structured numpy array must be named "
            "'index'. Adapt the user function."
        )
        assert len(data.dtype.names) > 1, (
            "The structured numpy array must have at least one additional "
            "field. Adapt the user function."
        )
        if name_of_output_property == "":
            raise RuntimeError(
                "You specified a JSON export format. In this case, you "
                "must also specify the 'name_of_output_property' in the "
                "configuration."
            )

        field_names = data.dtype.names

        # Convert the structured numpy array to a dictionary
        json_dump_data = {
            name_of_output_property: {
                str(entry[field_names[0]]): (
                    make_json_serializable(entry[field_names[1]])
                    if len(field_names) == 2
                    else [
                        make_json_serializable(entry[field])
                        for field in field_names[1:]
                    ]
                )
                for entry in data
            }
        }

        with open(output_file, "w") as json_file:
            try:
                json.dump(json_dump_data, json_file, indent=4)
            except TypeError as e:
                logging.error(f"Error writing JSON data: {e}")
                raise RuntimeError(
                    "Failed to write JSON data. Ensure all data is "
                    "JSON serializable."
                )

        return {name_of_output_property: data}
