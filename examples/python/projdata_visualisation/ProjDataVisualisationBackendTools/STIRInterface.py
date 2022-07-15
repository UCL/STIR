import time

import numpy
import stir
import stirextra

from enum import Enum, auto


class ProjdataDims(Enum):
    """
    Enum for the dimensions of a sinogram.
    """
    SEGMENT_NUM = auto()
    AXIAL_POS = auto()
    VIEW_NUMBER = auto()
    TANGENTIAL_POS = auto()


class ProjDataVisualisationBackend:
    """
    Class used as STIR interface to the projection data for ProjDataVisualisation.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Sets up the STIR interface for projection data.
        """
        print("ProjDataVisualisationBackend.__init__")

        self.proj_data_filename = ""
        self.proj_data_stream = None

        self.segment_data = None

        if len(args[0]) > 1:
            self.proj_data_filename = args[0][1]

        print("ProjDataVisualisationBackend.__init__: Done.")

    def load_projdata(self, filename=None) -> None:
        """
        Loads STIR projection data from a file.
        """
        if filename is not None:
            self.proj_data_filename = filename

        if self.proj_data_filename != "":
            print("ProjDataVisualisationBackend.load_data: Loading data from file: " + self.proj_data_filename)
            self.proj_data_stream = stir.ProjData_read_from_file(self.proj_data_filename)
            self.segment_data = self.refresh_segment_data()
            time.sleep(0.01)  # Wait for the log to be written...
            print("ProjDataVisualisationBackend.load_data: Data loaded.")
            self.print_proj_data_configuration()

    def print_proj_data_configuration(self) -> None:
        """
        Prints the configuration of the projection data.
        """
        print(
            f"\nProjection data configuration for:\n"
            f"\t'{self.proj_data_filename}'\n"
            f"\tNumber of views:\t\t\t\t\t{self.proj_data_stream.get_num_views()}\n"
            f"\tNumber of tangential positions:\t\t{self.proj_data_stream.get_num_tangential_poss()}\n"
            f"\tNumber of segments:\t\t\t\t\t{self.proj_data_stream.get_num_segments()}\n"
            f"\tNumber of axial positions:\t\t\t{self.proj_data_stream.get_num_axial_poss(0)}\n"
            f"\tNumber of tof positions:\t\t\t{self.proj_data_stream.get_num_tof_poss()}\n"
            f"\tNumber of non-tof sinograms:\t\t{self.proj_data_stream.get_num_non_tof_sinograms()}\n\n"
        )

    def print_segment_data_configuration(self) -> None:
        """
        Prints the configuration of the segment data.
        """
        print(
            f"\nSegment data configuration for:\n"
            f"\t'{self.proj_data_filename}':\n"
            f"\tSegment Number: {self.get_current_segment_num()}\n"
            f"\tNumber of views:\t\t\t\t\t{self.segment_data.get_num_views()}\n"
            f"\tNumber of tangential positions:\t\t{self.segment_data.get_num_tangential_poss()}\n"
            f"\tNumber of axial positions:\t\t\t{self.segment_data.get_num_axial_poss()}\n"
        )

    def refresh_segment_data(self, segment_number=0) -> stir.FloatSegmentByView:
        """
        Loads a segment data, from the projection data, into memory allowing for faster access.
        """
        if self.proj_data_stream is None:
            self.load_projdata()

        if self.proj_data_stream is not None:
            if self.segment_data is None:
                self.segment_data = self.proj_data_stream.get_segment_by_view(segment_number)

            elif segment_number != self.get_current_segment_num():
                self.segment_data = self.proj_data_stream.get_segment_by_view(segment_number)
            return self.segment_data

    @staticmethod
    def as_numpy(data: stir.ProjData) -> numpy.array:
        """
        Converts a STIR data object to a numpy array.
        """
        return stirextra.to_numpy(data)

    def get_limits(self, dimension: ProjdataDims, segment_number: int) -> tuple[int, int]:
        """
        Returns the limits of the projection data in the indicated dimension.
        :param dimension: The dimension to get the limits for, type SinogramDimensions.
        :param segment_number: The segment number to get the limits for. Only required for axial position.
        :return: A tuple containing the minimum and maximum value of the dimension (min, max).
        """
        if self.proj_data_stream is None:
            return (0, 0)

        if dimension == ProjdataDims.SEGMENT_NUM:
            return self.proj_data_stream.get_min_segment_num(), \
                   self.proj_data_stream.get_max_segment_num()
        elif dimension == ProjdataDims.AXIAL_POS:
            return self.proj_data_stream.get_min_axial_pos_num(self.get_current_segment_num()), \
                   self.proj_data_stream.get_max_axial_pos_num(self.get_current_segment_num())
        elif dimension == ProjdataDims.VIEW_NUMBER:
            return self.proj_data_stream.get_min_view_num(), \
                   self.proj_data_stream.get_max_view_num()
        elif dimension == ProjdataDims.TANGENTIAL_POS:
            return self.proj_data_stream.get_min_tangential_pos_num(), \
                   self.proj_data_stream.get_max_tangential_pos_num()
        else:
            raise ValueError("Unknown sinogram dimension: " + str(dimension))

    def get_num_indices(self, dimension: ProjdataDims):
        """
        Returns the number of indices in the given dimension.
        """
        if dimension == ProjdataDims.SEGMENT_NUM:
            return self.proj_data_stream.get_num_segments()
        elif dimension == ProjdataDims.AXIAL_POS:
            return self.proj_data_stream.get_num_axial_poss(self.get_current_segment_num())
        elif dimension == ProjdataDims.VIEW_NUMBER:
            return self.proj_data_stream.get_num_views()
        elif dimension == ProjdataDims.TANGENTIAL_POS:
            return self.proj_data_stream.get_num_tangential_poss()
        else:
            raise ValueError("Unknown sinogram dimension: " + str(dimension))

    def get_current_segment_num(self) -> int:
        """
        Returns the segment number of the current segment data.
        """
        if self.segment_data is not None:
            return self.segment_data.get_segment_num()
        return 0
