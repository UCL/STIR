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
        print(
            f"\nSegment data configuration for:\n"
            f"\t'{self.proj_data_filename}':\n"
            f"\tSegment Number: {self.segment_data.get_segment_num()}\n"
            f"\tNumber of views:\t\t\t\t\t{self.segment_data.get_num_views()}\n"
            f"\tNumber of tangential positions:\t\t{self.segment_data.get_num_tangential_poss()}\n"
            f"\tNumber of axial positions:\t\t\t{self.segment_data.get_num_axial_poss()}\n"
        )

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
