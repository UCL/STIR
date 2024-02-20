import numpy as np
import pytest

from examples.python.plot_projdata_profiles import compress_and_extract_1d_from_nd_array


def test_generate_1d_from_4d():
    """
    Test the generation of a 1D array from a 4D array.
    Primarily, this is a test of compress_and_extract_1d_from_nd_array that is used in plot_sinogram_profiles script.
    Given a 4D array, the function should compress the array to a 1D array based on the configuration provided.
    """
    np_4d = np.random.rand(2, 3, 4, 5)
    configs = [
        {
            "display_axis": 0,
            "projdata_indices": [None, None, None, None],
            "result_shape": (np_4d.shape[0],),
            "result_sum": np_4d.sum()
        },
        {
            "display_axis": 1,
            "projdata_indices": [None, None, None, None],
            "result_shape": (np_4d.shape[1],),
            "result_sum": np_4d.sum()
        },
        {
            "display_axis": 2,
            "projdata_indices": [None, None, None, None],
            "result_shape": (np_4d.shape[2],),
            "result_sum": np_4d.sum()
        },
        {
            "display_axis": 3,
            "projdata_indices": [None, None, None, None],
            "result_shape": (np_4d.shape[3],),
            "result_sum": np_4d.sum()
        },
        # Extracting a single value from certain dimensions
        # in some cases add an index in the display axis dimension to be ignored.
        {
            "display_axis": 0,
            "projdata_indices": [0, None, None, None],
            "result_shape": (np_4d.shape[0],),
            "result_sum": None
        },
        {
            "display_axis": 0,
            "projdata_indices": [0, 1, None, None],
            "result_shape": (np_4d.shape[0],),
            "result_sum": None
        },
        {
            "display_axis": 1,
            "projdata_indices": [None, 1, None, None],
            "result_shape": (np_4d.shape[1],),
            "result_sum": None
        },
        {
            "display_axis": 2,
            "projdata_indices": [None, None, 1, None],
            "result_shape": (np_4d.shape[2],),
            "result_sum": None
        },
        {
            "display_axis": 3,
            "projdata_indices": [None, None, None, 1],
            "result_shape": (np_4d.shape[3],),
            "result_sum": None
        },
    ]

    for config in configs:
        result = compress_and_extract_1d_from_nd_array(np_4d,
                                                       config["display_axis"],
                                                       config["projdata_indices"])
        assert result.shape == config["result_shape"]
        if config["result_sum"] is not None:
            pytest.approx(result.sum(), config["result_sum"], 1e-6)
