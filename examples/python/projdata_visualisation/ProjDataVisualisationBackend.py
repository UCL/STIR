import time

import numpy
import stir
import stirextra


class ProjDataVisualisationBackend:
    def __init__(self, *args, **kwargs):  # real signature unknown
        print("ProjDataVisualisationBackend.__init__")

        self.proj_data_filename = None
        self.proj_data_stream = None

        self.segment_data = None
        self.segment_number = None

        if len(args[0]) > 1:
            self.proj_data_filename = args[0][1]
            self.load_proj_data()

        print("ProjDataVisualisationBackend.__init__: Done.")

    def load_proj_data(self):
        """
        Loads STIR projection data from a file.
        """
        print("ProjDataVisualisationBackend.load_data: Loading data from file: " + self.proj_data_filename)
        self.proj_data_stream = stir.ProjData_read_from_file(self.proj_data_filename)
        time.sleep(0.01)
        print("ProjDataVisualisationBackend.load_data: Data loaded.")
        self.print_proj_data_configuration()

    def print_proj_data_configuration(self):
        print("ProjDataVisualisationBackend.print_proj_data_configuration:")
        print(
            f"\nProjection data configuration for:\n"
            f"\t'{self.proj_data_filename}':\n"
            f"\tNumber of views:\t\t\t\t\t{self.proj_data_stream.get_num_views()}\n"
            f"\tNumber of tangential positions:\t\t{self.proj_data_stream.get_num_tangential_poss()}\n"
            f"\tNumber of segments:\t\t\t\t\t{self.proj_data_stream.get_num_segments()}\n"
            f"\tNumber of axial positions:\t\t\t{self.proj_data_stream.get_num_axial_poss(0)}\n"
            f"\tNumber of tof positions:\t\t\t{self.proj_data_stream.get_num_tof_poss()}\n"
            f"\tNumber of non-tof sinograms:\t\t{self.proj_data_stream.get_num_non_tof_sinograms()}\n\n"
        )

    def print_segment_data_configuration(self):
        print("ProjDataVisualisationBackend.print_proj_data_configuration:")
        print(
            f"\nSegment data configuration for:\n"
            f"\t'{self.proj_data_filename}':\n"
            f"\tNumber of views:\t\t\t\t\t{self.proj_data_stream.get_num_views()}\n"
            f"\tNumber of tangential positions:\t\t{self.proj_data_stream.get_num_tangential_poss()}\n"
            f"\tNumber of segments:\t\t\t\t\t{self.proj_data_stream.get_num_segments()}\n"
            f"\tNumber of axial positions:\t\t\t{self.proj_data_stream.get_num_axial_poss(0)}\n"
            f"\tNumber of tof positions:\t\t\t{self.proj_data_stream.get_num_tof_poss()}\n"
            f"\tNumber of non-tof sinograms:\t\t{self.proj_data_stream.get_num_non_tof_sinograms()}\n\n"
        )

    # def check_range(self, dimension: str, value: int):
    #     """
    #     Checks if a value is within the range of the given dimension.
    #     """
    #     dimension = dimension.lower()
    #     if dimension == "axial":
    #         return 0 <= value < self.proj_data.get_num_axial_poss(0)
    #     elif dimension == "view":
    #         return 0 <= value < self.proj_data.get_num_views()
    #     elif dimension == "tangential":
    #         return 0 <= value < self.proj_data.get_num_tangential_poss()
    #     elif dimension == "segment":
    #         return 0 <= value < self.proj_data.get_num_segments()
    #     # elif dimension == "tof":
    #     #     return 0 <= value < self.proj_data.get_num_tof_poss()
    #     else:
    #         return False

    def get_sinogram(self, ax_pos_num, segment_num, as_numpy_array=False):
        """
        todo: REMOVE BECAUASE OLD
        Returns a sinogram for a given axial position and segment combination.
        """
        sinogram = self.proj_data_stream.get_sinogram(ax_pos_num, segment_num)
        if as_numpy_array:
            return stirextra.to_numpy(sinogram)
        return sinogram

    def get_viewgram(self, view_num, segment_num, as_numpy_array=False):
        """
        todo: REMOVE BECAUASE OLD
        Returns a viewgram for a given view and segment combination.
        """
        viewgram = self.proj_data_stream.get_viewgram(view_num, segment_num)
        if as_numpy_array:
            return stirextra.to_numpy(viewgram)
        return viewgram

    def refresh_segment_data(self, segment_number=0):
        """
        Loads a segment from the projection data.
        """
        if segment_number != self.segment_number:
            self.segment_number = segment_number
            self.segment_data = self.proj_data_stream.get_segment_by_view(segment_number)
        return self.segment_data

    def as_numpy(self, data: stir.ProjData) -> numpy.array:
        """
        Converts a STIR data object to a numpy array.
        """
        return stirextra.to_numpy(data)
