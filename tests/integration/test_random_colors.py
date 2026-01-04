"""Integration test for i2pp."""

import json
import os
import subprocess
import tempfile

import jinja2


def test_i2pp_integration_random_colors():
    """Test that i2pp generates the expected output for a sample image data set
    with random colors.

    The corresponding mesh is a 5x2x10 mesh.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    input_file = os.path.join(
        current_dir, "test_data/random_colors.yaml.jinja"
    )
    expected_output = os.path.join(
        current_dir, "test_data/expected_physical_property.json"
    )
    image_folder = os.path.join(current_dir, "test_data/image_data")
    mesh_file = os.path.join(current_dir, "test_data/5x2x10_mesh.4C.yaml")
    user_function = os.path.join(
        current_dir, "test_data/test_user_function.py"
    )

    # Temporary output file
    with tempfile.TemporaryDirectory() as tmpdir:
        # take the input file and render it with Jinja2
        with open(input_file, "r") as file:
            content = file.read()
        template = jinja2.Template(content)
        rendered_content = template.render(
            tmp_test_dir=tmpdir,
            mesh_file=mesh_file,
            image_folder=image_folder,
            user_function=user_function,
        )

        input_file_filled = os.path.join(tmpdir, "random_colors.yaml")
        with open(input_file_filled, "w") as file:
            file.write(rendered_content)

        output_file = os.path.join(tmpdir, "physical_property.json")

        # Run the i2pp tool via subprocess
        result = subprocess.run(["i2pp", input_file_filled])

        # Check that the command ran successfully
        assert result.returncode == 0, f"i2pp failed: {result.stderr}"

        # Check that the output file exists
        assert os.path.isfile(output_file), "Output file does not exist."

        # Open and load the JSON file
        with open(output_file, "r") as file:
            data = json.load(file)

        # Check that the output file is not empty
        assert data, "Output file is empty."

        # Open and load the expected output file
        with open(expected_output, "r") as file:
            expected_data = json.load(file)

        # Check that the output data matches the expected data
        assert data == expected_data, (
            f"Output data does not match expected data.\n"
            f"Output: {data}\nExpected: {expected_data}"
        )

        # print data
        print("Output data:", data)
