import matplotlib.pyplot as plt
from os import getcwd
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
    axs.legend(["Mean", "Median"])
    axs.title.set_text(
        f"{pretitle} l2-norm of distance to origin: Mean = {round_sig(mean)} and Median = {round_sig(median)}")


class ViewOffsetConsistencyClosestLORInfo:
    """
    Helper class.
    Loads and converts the data outputs by the `test_view_offset_root` test in STIR into coordinates.
    The first line of the text file should be the original coordinate of the point source and
    the rest the closest voxel along the LOR to the original.
    Each line is expected to be formatted as [ x y z ], e.g., '190.048 0 145.172\n'
    """

    def __init__(self, filename, tolerance=6.66983):
        """
        :param filename: Filename of the file output by `test_view_offset_root` (.txt file)
        :param tolerance: l2-norm tolerance that classified a "failed" event. Default = 6.66983, from previous studies.
        """
        self.filename = filename
        print(f"Loading data: {self.filename}")
        with open(filename) as f:
            self.lines = f.readlines()
        self.original_coord, self.coord_list = self.__extract_coords_from_lines(self.lines)
        self.err_l2 = self.compute_distance_to_original()
        self.err_x = self.compute_distance_to_original(0)
        self.err_y = self.compute_distance_to_original(1)
        self.err_z = self.compute_distance_to_original(2)

        self.tolerance = tolerance

    def __process_line(self, line):
        """
        Given a line (e.g., '190.048 0 145.172\n'), strips the " " and "\n" and returns the coordinate as a numpy array
        :param line: string input from a line in the file.
        :return: numpy array of (x,y,z)
        """
        coord = line[:-2].split(" ")
        return np.array([float(coord[0]), float(coord[1]), float(coord[2])])

    def __extract_coords_from_lines(self, lines):
        """
        Assumes that lines comes in as a string with three numbers, each split by a space.
        Iterates through each file to get coordinate data for each entry.
        First line is original coordinate of the point source and
        following lines are the closest voxel to the origin for each LOR in the root file.
        :param lines: Input lines, loaded from the file
        :return: A tuple of the original coordinate numpy array and a list of coordinates of closes voxels in the LOR
        """
        is_first = True
        coord_list = []
        for line in lines:
            if is_first:
                original_coord = self.__process_line(line)
                is_first = False
                continue
            coord_list.append(self.__process_line(line))
        return (original_coord, coord_list)

    def compute_distance_to_original(self, axis=None):
        if axis is None:
            # Compute l2
            return [np.linalg.norm(self.original_coord - coord) for coord in self.coord_list]
        else:
            # compute distance in 1 dimension, but extends negative
            return [coord[axis] - self.original_coord[axis] for coord in self.coord_list]

    def get_num_events(self):
        return len(self.coord_list)

    def get_num_failed_events(self, tolerance=None):
        if tolerance is not None:
            print(f"Overwriting tolerance value as {tolerance}")
            self.tolerance = tolerance
        return sum(errs > self.tolerance for errs in self.err_l2)


def print_pass_and_fail(allowed_fraction_of_failed_events=0.5):
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


def print_axis_biases():
    """
    Print the mean offset in each axis (x,y,z) for each point source file and the total bias in each axis
    :return: None
    """

    total_bias_x = 0
    total_bias_y = 0
    total_bias_z = 0
    string_length = 10

    # Construct the table header
    header_entries = ["SourceID", "Mean Offset (x)", "Mean Offset (y)", "Mean Offset (z)"]
    string_length = 2 + max([len(entry) for entry in header_entries])
    header_string = table_row_as_string(header_entries, string_length)
    # Print the table header
    print(f"\nMean offset in each axis for each source position\n"
          f"{header_string}\n"
          f"{'-' * len(header_string)}")

    # Loop over each point source and print the mean offset in each axis
    # Also compute the total bias in each axis
    for key in point_sources_data.keys():
        mean_err_x = np.mean(point_sources_data[key].err_x)
        mean_err_y = np.mean(point_sources_data[key].err_y)
        mean_err_z = np.mean(point_sources_data[key].err_z)
        total_bias_x += mean_err_x
        total_bias_y += mean_err_y
        total_bias_z += mean_err_z
        row = [key, round_sig(mean_err_x, 3), round_sig(mean_err_y, 3), round_sig(mean_err_z, 3)]
        row_string = table_row_as_string(row, string_length)
        print(f"{row_string}")

    # Print the total bias, the mean of aforementioned offsets for each point sources axis.
    # This should only raise alarm if one of these numbers is large.

    # Construct the table header
    num_entries = len(point_sources_data.keys())
    header_entries = ["Total Bias (x)", "Total Bias (y)", "Total Bias (z)"]
    string_length = 2 + max([len(entry) for entry in header_entries])
    header_string = table_row_as_string(header_entries, string_length)

    # Print the table header
    print(f"\nTOTAL BIAS IN EACH AXIS\n"
          f"\n{header_string}\n"
          f"{'-' * len(header_string)}")
    row = [round_sig(total_bias_x / num_entries, 3),
           round_sig(total_bias_y / num_entries, 3),
           round_sig(total_bias_z / num_entries, 3)]
    row_string = table_row_as_string(row, string_length)
    print(f"{row_string}")


print("\nUSAGE: After `make test` or `test_view_offset_root` has been run,\n"
      "run `debug_view_offset_consistency` from `pretest_output` directory or input that directory as an argument.\n")

working_directory = getcwd()
if len(sys.argv) > 1:
    working_directory = sys.argv[1]

point_sources_data = dict()
filename_prefix = "root_header_test"
filename_suffix = "_lor_pos.txt"

for i in range(1, 12, 1):
    point_sources_data[i] = ViewOffsetConsistencyClosestLORInfo(f"{filename_prefix}{i}{filename_suffix}")
    # plot_1D_distance_histogram(point_sources_data[i].err_l2, pretitle=f"Point {i}:")

print_pass_and_fail()
print_axis_biases()

print("Done")
