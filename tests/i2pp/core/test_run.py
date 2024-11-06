"""Test run procedure."""

from unittest.mock import MagicMock, patch

from i2pp.core.run import run_i2pp
from munch import munchify


def test_run_i2pp() -> None:
    """Test run procedure of img2physiprop."""

    mock_config = munchify({"key": "value"})

    mock_run_manager = MagicMock()

    with patch("i2pp.core.run.RunManager", return_value=mock_run_manager):
        mock_exemplary_function = MagicMock(return_value="Exemplary output")
        with patch(
            "i2pp.core.run.exemplary_function", mock_exemplary_function
        ):
            run_i2pp(mock_config)

    mock_run_manager.init_run.assert_called_once()
    mock_exemplary_function.assert_called_once()
    mock_run_manager.finish_run.assert_called_once()
