import csv

import matplotlib.pyplot as plt
from os import chdir
import sys
import numpy as np
from math import log10, floor


def round_sig(x, sig=2):
    """
    Utility function to convert a float into a float with sig significant figures
    :param x: input float
    :param sig: number of significant figures
    :return: float that is rounded to a number of significant figures
    """
    if x == 0.0:
        return 0.0  # Avoid log(0)
    return round(x, sig - int(floor(log10(abs(x)))) - 1)


def pad_string(s, length=10, pad_char=" "):
    """
    Pads a string to a certain length with a certain character
    :param s: variable to be covnerted to a string and padded on either side
    :param length: Length of padded string
    :param pad_char: Character to pad with (assumes length 1)
    :return: Padded string
    """
    s = str(s)
    assert len(pad_char) == 1

    total_padding = max(length - len(s), 0)
    if (length - len(s)) % 2 == 0:
        # Pad equally on both sides
        padding = " " * (total_padding // 2)
        return padding + s + padding
    else:
        # Pad unevenly on both sides (prefer right)
        right_padding = pad_char * (total_padding // 2)
        left_padding = pad_char * (total_padding // 2 + 1)
        return left_padding + s + right_padding


def table_row_as_string(row, entry_length):
    """
    Constructs a string from a list of entries, each entry is padded to a certain length
    :param row: List of entries to be converted to a string
    :param entry_length: Length of each entry
    :return: String of row entries with padding
    """
    return "|".join([pad_string(entry, entry_length) for entry in row])


def plot_1D_distance_histogram(distances, n_bins=100, logy=False, pretitle=""):
    """
    Plot distances to original on a 1D histogram
    :param distances: distances to origin
    :param n_bins: Number of histogram bins
    :param logy: Use log in y axis
    :param pretitle: prefix for title, useful for indicating point source number etc.
    :return: None
    """
    fig = plt.figure(figsize=(15, 5))
    # Subplot 1
    axs = fig.add_subplot()
    axs.hist(distances, bins=n_bins, log=logy)
    mean = np.mean(distances)
    median = np.median(distances)
    axs.axvline(mean, color='r')
    axs.axvline(median, color='g')
    plt.ylabel("Number of events")
    plt.xlabel("Distance to original (mm)")
    axs.legend(["Mean", "Median"])
    axs.title.set_text(
        f"{pretitle} l2-norm of distance to origin: Mean = {round_sig(mean)} and Median = {round_sig(median)}")


def point_cloud_3D(data_handler):
    """
    Plot the pointcloud of generated LOR-positions, original source-position and tolerance-whiskers for each source in a seperate scatter plot.
    Args:
        data_handler (ROOTConsistencyDataHandler): Object containing all relevant data for the plot
    """

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    # Plot all the points (intensity increases for multiple points)
    ax.scatter(data_handler.voxel_coords[:, 0], data_handler.voxel_coords[:, 1], data_handler.voxel_coords[:, 2])

    # Plot the original point and tolerance
    ox = data_handler.original_coord[0]
    oy = data_handler.original_coord[1]
    oz = data_handler.original_coord[2]
    tol = data_handler.tolerance
    ax.plot(ox, oy, oz, c='r', marker='o')
    # plot tolerence around original point
    ax.plot([ox + tol, ox - tol], [oy, oy], [oz, oz], c='r', marker="_", label='_nolegend_')
    ax.plot([ox, ox], [oy + tol, oy - tol], [oz, oz], c='r', marker="_", label='_nolegend_')
    ax.plot([ox, ox], [oy, oy], [oz + tol, oz - tol], c='r', marker="_", label='_nolegend_')

    # plot Mean position and standard deviation
    fx = data_handler.mean_coord[0]
    fy = data_handler.mean_coord[1]
    fz = data_handler.mean_coord[2]
    xerror = np.std(data_handler.voxel_coords[:, 0])
    yerror = np.std(data_handler.voxel_coords[:, 1])
    zerror = np.std(data_handler.voxel_coords[:, 2])
    ax.plot(fx, fy, fz, linestyle="None", marker="o", c='g')
    ax.plot([fx + xerror, fx - xerror], [fy, fy], [fz, fz], marker="_", c='g', label='_nolegend_')
    ax.plot([fx, fx], [fy + yerror, fy - yerror], [fz, fz], marker="_", c='g', label='_nolegend_')
    ax.plot([fx, fx], [fy, fy], [fz + zerror, fz - zerror], marker="_", c='g', label='_nolegend_')

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    ax.legend(['Voxel Positions', 'Origin and Tolerance', 'Mean coords and stddev'])

    plt.show()


def point_cloud_3D_all(dict_of_data_handlers):
    """
    generates a single plot with pointclouds of generated LOR-positions, original source-positions and tolerance-whiskers for all sources
    Args:
        dict_of_data_handlers (dictionary containing objects of type ROOTConsistencyDataHandler): dictionary of objects containing all relevant data for the plot
    """

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    for key in dict_of_data_handlers.keys():
        data_handler = dict_of_data_handlers[key]

        # Plot all the points (intensity increases for multiple points)
        ax.scatter(data_handler.voxel_coords[:, 0], data_handler.voxel_coords[:, 1], data_handler.voxel_coords[:, 2],
                   label='Source' + str(key))

        # Plot the original point and tolerance
        ox = data_handler.original_coord[0]
        oy = data_handler.original_coord[1]
        oz = data_handler.original_coord[2]
        tol = data_handler.tolerance
        ax.plot(ox, oy, oz, c='r', marker='o')
        # plot tolerence around original point
        ax.plot([ox + tol, ox - tol], [oy, oy], [oz, oz], c='r', marker="_", label='_nolegend_')
        ax.plot([ox, ox], [oy + tol, oy - tol], [oz, oz], c='r', marker="_", label='_nolegend_')
        ax.plot([ox, ox], [oy, oy], [oz + tol, oz - tol], c='r', marker="_", label='_nolegend_')

        # plot Mean position and standard deviation
        fx = data_handler.mean_coord[0]
        fy = data_handler.mean_coord[1]
        fz = data_handler.mean_coord[2]
        xerror = np.std(data_handler.voxel_coords[:, 0])
        yerror = np.std(data_handler.voxel_coords[:, 1])
        zerror = np.std(data_handler.voxel_coords[:, 2])
        ax.plot(fx, fy, fz, linestyle="None", marker="o", c='g')
        ax.plot([fx + xerror, fx - xerror], [fy, fy], [fz, fz], marker="_", c='g', label='_nolegend_')
        ax.plot([fx, fx], [fy + yerror, fy - yerror], [fz, fz], marker="_", c='g', label='_nolegend_')
        ax.plot([fx, fx], [fy, fy], [fz + zerror, fz - zerror], marker="_", c='g', label='_nolegend_')

        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlabel('z (mm)')

        # set the x-spine
        ax.spines['left'].set_position('zero')
        # set the y-spine
        ax.spines['bottom'].set_position('zero')

    plt.legend()
    plt.show()


def __extract_data_from_csv_file(filename):
    """
    Loads data from a csv file and returns a numpy array containing the data.
    Assumes first column is the row's key.
    Each row is a separate entry in the array.
    Generally, the first row is the tolerance and the second row is the original point.
    Remainder of rows are the voxel positions of measured data.
    :param filename: Name of the csv file to load the data from.
    :return: A tuple of the original coordinate numpy array and a list of coordinates of closes voxels in the LOR
    """

    # Read the file and extract the csv data
    with open(filename, newline='') as csvfile:
        data = list(csv.reader(csvfile))

    # Setup the tolerance and arrays for the data
    tolerance = None
    original_coord = np.zeros(shape=3)
    entry_coords = np.zeros(shape=(len(data) - 2, 3))  # -2 because first 2 lines are tolerance and original coord

    line_index = 0
    for entry in data:
        if entry[0] == "tolerance":
            tolerance = float(entry[1])
        elif entry[0] == "original coordinates":
            original_coord = np.array([float(entry[1]), float(entry[2]), float(entry[3])])
        else:
            entry_coords[line_index] = np.array([float(entry[1]), float(entry[2]), float(entry[3])])
            line_index += 1
    return original_coord, entry_coords, tolerance


class ROOTConsistencyDataHandler:
    """
    Helper class.
    Loads and converts the data output by the `test_consistency_with_root` test.
    The first line of the text file should be the original coordinate of the point source and
    the rest correspond to a selected voxel position along the LOR.
    Each line is expected to be formatted as [ x y z ], e.g., '190.048 0 145.172\n'
    """

    def __init__(self, filename):
        """
        :param filename: Filename of the file output by `test_view_offset_GATE` (.csv file)
        """
        self.filename = filename
        print(f"Loading data: {self.filename}")

        # Extract the original coordinate and voxel coordinates as 2D numpy arrays
        self.original_coord, self.voxel_coords, self.tolerance = __extract_data_from_csv_file(filename)

        if self.tolerance is None:
            raise Exception("No tolerance information found in file")

        if self.voxel_coords.size == 0:
            raise Exception("No voxel coordinates found in file")

        # Mean coordinate of the lor voxels voxels
        self.mean_coord = np.mean(self.voxel_coords, axis=0)

        self.voxel_offset = self.voxel_coords - self.original_coord
        self.entrywise_l2_norm = np.linalg.norm(self.voxel_offset, axis=1)
        self.l2 = np.linalg.norm(self.voxel_offset)

    def get_num_events(self):
        return len(self.voxel_coords)

    def set_tolerance(self, tolerance):
        print(f"Overriding tolerance value as {tolerance}")
        self.tolerance = tolerance

    def get_num_failed_events(self, tolerance=None):
        """
        Returns the number of events that are outside the tolerance
        :param tolerance: l2-norm tolerance that classified a "failed" event.
        """
        if tolerance is None:
            tolerance = self.tolerance
        return len([err for err in self.entrywise_l2_norm if err > tolerance])

    def get_failure_percentage(self, tolerance=None):
        """
        Returns the percentage of events that are outside the tolerance
        :param tolerance: l2-norm tolerance that classified a "failed" event.
        """
        return self.get_num_failed_events(tolerance) / self.get_num_events()


def print_pass_and_fail(point_sources_data, allowed_fraction_of_failed_events=0.05):
    """
    Prints the number of events, how many events failed and the percentage
    :param allowed_fraction_of_failed_events: Fraction of events that could "fail" before the test failed
    :return: None
    """

    # Construct Table Header
    header_entries = ["SourceID", "Number of Events", "Failed Events", "Failure (%)"]
    string_length = 2 + max([len(col) for col in header_entries])
    header_string = table_row_as_string(header_entries, string_length)

    # Print table header
    print(f"\nInformation regarding pass fail rate of the view offset test\n"
          f"{header_string}\n"
          f"{'-' * len(header_string)}")

    # Loop over each point source and print the number of events, number of failed events and failure percentage
    for key in point_sources_data.keys():
        num_events = point_sources_data[key].get_num_events()
        num_failed_events = point_sources_data[key].get_num_failed_events()
        percentage = num_failed_events / num_events * 100
        row = [key, num_events, num_failed_events, round_sig(percentage, 3)]
        row_string = table_row_as_string(row, string_length)
        if percentage > allowed_fraction_of_failed_events * 100:
            warning_msg = "HIGH VALUE WARNING!"
        else:
            warning_msg = ""
        print(f"{row_string} {warning_msg}")


def print_axis_biases(point_sources_data):
    """
    Prints the LOR COM and the axis bias for each point source and the overall bias in each axis.
    :return: None
    """

    total_bias = np.zeros(shape=(3,))

    # Construct the table header
    header_entries = ["SourceID", "x (offset)", "y (offset)", "z (offset)"]
    # Loop over each point source and print the mean offset in each axis
    # Also compute the total bias in each axis
    row_entries = dict()
    for key in point_sources_data.keys():
        x = point_sources_data[key].mean_coord[0]
        y = point_sources_data[key].mean_coord[1]
        z = point_sources_data[key].mean_coord[2]
        err_x = np.mean(point_sources_data[key].voxel_offset[:, 0])
        err_y = np.mean(point_sources_data[key].voxel_offset[:, 1])
        err_z = np.mean(point_sources_data[key].voxel_offset[:, 2])

        # Construct an row of entries for the table: e.g., ['1', '188.0 (-0.361)', '0.61 (-0.361)','146.0 (-0.361)']
        row_entries[key] = [str(key),
                            f"{round_sig(x, 3)} ({round_sig(err_x, 3)})",
                            f"{round_sig(y, 3)} ({round_sig(err_y, 3)})",
                            f"{round_sig(z, 3)} ({round_sig(err_z, 3)})"]

        # Add the bias to the total bias
        total_bias += np.array([err_x, err_y, err_z])

    # Get the max entry length of both the heading entries or row entries
    string_length = 2 + max(max([len(entry) for entry in header_entries]),
                            max([max([len(i) for i in entry]) for entry in row_entries.values()]))

    # Print the table header row
    header_string = table_row_as_string(header_entries, string_length)
    print(f"\nMean closest lor voxel position for each data set in each axis (with offset) \n"
          f"{header_string}\n"
          f"{'-' * len(header_string)}")
    # Iteratively print the table rows
    for key in row_entries.keys():
        row_string = table_row_as_string(row_entries[key], string_length)
        print(f"{row_string}")

    # Print the total bias, the mean of aforementioned offsets for each point sources axis.
    # This should only raise alarm if one of these numbers is large.

    # Construct the table header
    num_entries = len(point_sources_data.keys())
    header_entries = ["Total Bias (x)", "Total Bias (y)", "Total Bias (z)"]
    string_length = 2 + max([len(entry) for entry in header_entries])
    header_string = table_row_as_string(header_entries, string_length)

    # Print the table header
    print(f"\nTOTAL BIAS IN EACH AXIS"
          f"\n{header_string}\n"
          f"{'-' * len(header_string)}")
    row = [round_sig(total_bias[0] / num_entries, 3),
           round_sig(total_bias[1] / num_entries, 3),
           round_sig(total_bias[2] / num_entries, 3)]
    row_string = table_row_as_string(row, string_length)
    print(f"{row_string}")


def nonTOF_evaluation(filename_prefix, file_extension=".csv"):
    # Loop over all files in the working directory and load the data into the point_sources_data dictionary
    point_sources_data = dict()
    for i in range(1, 9, 1):
        point_sources_data[i] = ROOTConsistencyDataHandler(f"{filename_prefix}{i}{file_extension}")

    # Print the number of events, number of failed events and failure percentage for each point source
    print_pass_and_fail(point_sources_data)
    # Print the mean offset in each axis (x,y,z) for each point source and the total bias in each axis
    print_axis_biases(point_sources_data)


def TOF_evaluation(filename_prefix, file_extension=".csv"):
    # Loop over all files in the working directory and load the data into the point_sources_data dictionary
    point_sources_data = dict()
    for i in range(1, 9, 1):
        point_sources_data[i] = ROOTConsistencyDataHandler(f"{filename_prefix}{i}{file_extension}")

    # Print the number of events, number of failed events and failure percentage for each point source
    print_pass_and_fail(point_sources_data)
    # Print the mean offset in each axis (x,y,z) for each point source and the total bias in each axis
    print_axis_biases(point_sources_data)

    point_cloud_3D(point_sources_data[1])
    point_cloud_3D_all(point_sources_data)


# =====================================================================================================
# Main Script
# =====================================================================================================
def main():
    print("\nUSAGE: After `make test` or `test_view_offset_GATE` has been run,\n"
          "run `debug_view_offset_consistency` from `ROOT_STIR_consistency` directory or input that directory as an "
          "argument.\n")

    # Optional argument to set the directory of the output of the test_consistency_with_root CTest.
    if len(sys.argv) > 1:
        chdir(sys.argv[1])

    nonTOF_evaluation("non_TOF_voxel_data_", file_extension=".csv")

    TOF_evaluation("TOF_voxel_data_", file_extension=".csv")

    print("Done")


if __name__ == "__main__":
    main()
