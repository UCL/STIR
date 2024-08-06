# Demo to plot the profile of projection data using STIR
# To run in "normal" Python, you would type the following in the command line
#  execfile('plot_projdata_profiles.py')
# In ipython, you can use
#  %run plot_projdata_profiles.py
# Or of course
#  import plot_projdata_profiles

# Copyright 2021 University College London
# Copyright 2024 Prescient Imaging

# Authors: Robert Twyman

# This file is part of STIR.
# SPDX-License-Identifier: Apache-2.0
# See STIR/LICENSE.txt for details

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

import stir
import stirextra

PROJDATA_DIM_MAP = {
    0: "TOF",
    1: "Axial Segment",
    2: "View",
    3: "Tangential"
}


def get_projdata_from_file_as_numpy(filename: str) -> np.ndarray | None:
    """
    Load a Projdata file and convert it to a NumPy array.
    Args:
        filename: The filename of the Projdata file to load.
    Returns:
        result: The NumPy array.
    """
    try:
        projdata: stir.ProjData = stir.ProjData.read_from_file(filename)
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return None

    try:
        return stirextra.to_numpy(projdata)
    except Exception as e:
        print(f"Error converting to numpy: {e}")
        return None


def get_projection_data_as_array(f: str | stir.ProjData) -> np.ndarray | None:
    """
    Get the projection data from a file or object.
    Args:
        f: The file name or object to get the projection data from.
    Returns:
        result: The projection data as a NumPy array.
    """
    # Get the input data from a file or object
    if isinstance(f, str):
        print(f"Handling:\n\t{f}")
        return get_projdata_from_file_as_numpy(f)

    elif isinstance(f, stir.ProjData):
        try:
            return stirextra.to_numpy(f)
        except AttributeError as e:
            print(f"AttributeError converting to projdata to numpy.\nError message{e}")
            return None

    else:
        print(f"Unknown type for {f=}")
        return None


def compress_and_extract_1d_from_nd_array(data: np.ndarray,
                                          display_axis: int,
                                          axes_indices: list[int | None] | None = None
                                          ) -> np.ndarray:
    """
    Generate a 1D array from an n-dimensional NumPy array based on specified parameters.
    The display is the axis to be extracted.
    The axes_indices is a list of indices to extract from each dimension.
    If the index is None, the entire dimension is summed.
    If the index is not None, the data is taken from that index.
    Args:
        data: The n-dimensional NumPy array.
        display_axis: The index of the dimension to be treated as the horizontal component.
        axes_indices: A list of indices to extract from each dimension.
            If None, all indices, except the display axis, are summed.
    Returns:
        result: The 1D NumPy array.
    Exceptions:
        ValueError: If the data is not at least 2D.
        ValueError: If the number of axes indices does not match the number of dimensions.
        ValueError: If the indices are out of bounds.
    """
    if data.ndim < 2:
        raise ValueError(f"Data must have at least 2 dimensions, not {data.ndim}D")

    if axes_indices is None:
        axes_indices = [None] * data.ndim
    if not len(axes_indices) == data.ndim:
        raise ValueError(
            f"Number of axes indices ({len(axes_indices)}) must match the number of dimensions ({data.ndim})")

    working_axis = 0
    # Check if indices are within valid range for all dimensions
    for data_axis, index in enumerate(axes_indices):
        if index is not None and not np.all(np.logical_and(index >= 0, index < data.shape[data_axis])):
            raise ValueError(f"Indices for axis {data_axis} are out of bounds. {index=}, {data.shape[data_axis]=}")

    for data_axis in range(data.ndim):
        if display_axis == data_axis:
            working_axis += 1
        elif axes_indices[data_axis] is None:
            data = np.sum(data, axis=working_axis)
        else:
            data = np.take(data, axes_indices[data_axis], axis=working_axis)
    return data


def plot_projdata_profiles(projection_data_list: list[stir.ProjData] | list[str],
                           display_axis: int = 3,
                           data_indices: list[int | None] | None = None,
                           ) -> None:
    """
    Plots the profiles of the projection data.
    Compress (via sum) and extract a 1D array from a 4D array of projection data for each element of the list.
    Args:
        projection_data_list: list of projection data file names or stir.ProjData objects to load and plot.
        display_axis: The horizontal component of the projection data to plot.
        data_indices: The indices to extract from the projection data (None indices are summed).
    Returns:
        None
    """

    plt.figure()
    ax = plt.subplot(111)

    for f in projection_data_list:
        if isinstance(f, str):
            label = f
        else:
            label = ""

        projdata_npy = get_projection_data_as_array(f)
        if projdata_npy is None:
            continue

        # Generate the 1D array
        try:
            plot_data = compress_and_extract_1d_from_nd_array(projdata_npy, display_axis, data_indices)
        except ValueError as e:
            print(f"Error generating 1D array object.\nError message: {e}")
            continue

        plt.plot(plot_data, label=label)

    if len(plt.gca().get_lines()) == 0:
        print("Something went wrong! No data to plot.")
        return

    # Identify sum and extraction axes
    sum_axis = [i for i, x in enumerate(data_indices) if x is None and i != display_axis]
    index_axis = [i for i, x in enumerate(data_indices) if x is not None and i != display_axis]

    # Extract labels and values for sum and extraction axes
    sum_axis_labels = [PROJDATA_DIM_MAP[i] for i in sum_axis]
    extraction_axis_labels = [PROJDATA_DIM_MAP[i] for i in index_axis]
    index_values = [data_indices[i] for i in index_axis]

    # Plot title
    plt.title(f"Summing {sum_axis_labels} axis and extracting {extraction_axis_labels} with values {index_values}")
    plt.xlabel(f"{PROJDATA_DIM_MAP[display_axis]}")
    ax.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.description = ("This script loads, sums axis' and plots profiles over input projection data files."
                          "The default is to sum over all components, except the display axis."
                          "The indices used are array based, not STIR offset based.")
    parser.add_argument('filenames',
                        nargs='*',
                        help='Projection data file names to show, can handle multiple.')
    parser.add_argument('--display_axis',
                        dest="display_axis",
                        type=int,
                        default=3,
                        help='The horizontal component of the projection data to plot.'
                             'The default is -1 indicating a sum over all components. '
                             '0: TOF, 1: axial (and segment), 2: view, 3: tangential.')
    parser.add_argument('--tof',
                        dest="tof",
                        type=int,
                        default=None,
                        help='The TOF value of the projection data to plot.'
                             'The default is to sum over all TOF values.')
    parser.add_argument('--axial_segment',
                        dest="axial_segment",
                        type=int,
                        default=None,
                        help='The axial segment number of the projection data to plot.'
                             'The default is to sum over all axial segments.')
    parser.add_argument('--view',
                        dest="view",
                        type=int,
                        default=None,
                        help='The view of the projection data to plot.'
                             'The default is to sum over all views.')
    parser.add_argument('--tangential_pos',
                        dest="tangential",
                        type=int,
                        default=None,
                        help='The tangential position of the projection data to plot.'
                             'The default is to sum over all tangential positions.')

    args = parser.parse_args()

    if len(args.filenames) < 1:
        parser.print_help()
        exit(0)

    plot_projdata_profiles(projection_data_list=args.filenames,
                           display_axis=args.display_axis,
                           data_indices=[args.tof, args.axial_segment, args.view, args.tangential]
                           )
