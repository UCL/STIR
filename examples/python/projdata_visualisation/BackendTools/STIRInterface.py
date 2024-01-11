# Copyright 2022, 2024 University College London

# Author Robert Twyman

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

import time

import numpy
import stir
import stirextra

from enum import Enum, auto


class ProjDataDims(Enum):
    """Enum for the dimensions of a sinogram."""
    SEGMENT_NUM = auto()
    AXIAL_POS = auto()
    VIEW_NUMBER = auto()
    TANGENTIAL_POS = auto()
    TIMING_POS = auto()

class ProjDataVisualisationBackend:
    """Class used as STIR interface to the projection data for ProjDataVisualisation."""
    def __init__(self, *args, **kwargs) -> None:
        """Sets up the STIR interface for projection data."""

        self.projdata_filename = ""
        self.projdata = None

        self.segment_data = None

    def load_projdata(self, filename=None) -> bool:
        """Loads STIR projection data from a file."""
        if filename is not None and filename != "":
            self.projdata_filename = filename

        if self.projdata_filename != "":
            print("ProjDataVisualisationBackend.load_data: Loading data from file: " + self.projdata_filename)
            try:
                self.projdata = stir.ProjData.read_from_file(self.projdata_filename)
                self.segment_data = self.refresh_segment_data()
                print("ProjDataVisualisationBackend.load_data: Data loaded.")
                self.print_projdata_configuration()
            except RuntimeError:
                print("ProjDataVisualisationBackend.load_data: Error loading data from file: " + self.projdata_filename)
                return False
            return True
        return False
        
    def set_projdata(self, projdata: stir.ProjData) -> None:
        """Sets the projection data stream."""
        self.projdata = projdata
        self.projdata_filename = "ProjData filename set externally, no filename"
        self.segment_data = self.refresh_segment_data()
        self.print_projdata_configuration()

    def print_projdata_configuration(self) -> None:
        """Prints the configuration of the projection data."""
        print(
            f"\nProjection data configuration for:\n"
            f"\t'{self.projdata_filename}'\n"
            f"\tNumber of views:\t\t\t\t\t{self.projdata.get_num_views()}\n"
            f"\tNumber of tangential positions:\t\t{self.projdata.get_num_tangential_poss()}\n"
            f"\tNumber of segments:\t\t\t\t\t{self.projdata.get_num_segments()}\n"
            f"\tNumber of axial positions:\t\t\t{self.projdata.get_num_axial_poss(0)}\n"
            f"\tNumber of tof positions:\t\t\t{self.projdata.get_num_tof_poss()}\n"
            f"\tNumber of non-tof sinograms:\t\t{self.projdata.get_num_non_tof_sinograms()}\n\n"
        )

    def print_segment_data_configuration(self) -> None:
        """Prints the configuration of the segment data."""
        print(
            f"\nSegment data configuration for:\n"
            f"\t'{self.projdata_filename}':\n"
            f"\tSegment Number: {self.get_current_segment_num()}\n"
            f"\tNumber of views:\t\t\t\t\t{self.segment_data.get_num_views()}\n"
            f"\tNumber of tangential positions:\t\t{self.segment_data.get_num_tangential_poss()}\n"
            f"\tNumber of axial positions:\t\t\t{self.segment_data.get_num_axial_poss()}\n"
        )

    def refresh_segment_data(self, segment_number=0, timing_pos=0) -> stir.FloatSegmentByView:
        """Loads a segment data, from the projection data, into memory allowing for faster access."""
        if self.projdata is None:
            self.load_projdata()

        if self.projdata is not None:
            seg_idx = stir.SegmentIndices(segment_number, timing_pos)
            self.segment_data = self.projdata.get_segment_by_view(seg_idx)
            return self.segment_data

    @staticmethod
    def as_numpy(data: stir.ProjData) -> numpy.array:
        """Converts a STIR data object to a numpy array."""
        return stirextra.to_numpy(data)

    def get_limits(self, dimension: ProjDataDims, segment_number: int) -> tuple:
        """
        Returns the limits of the projection data in the indicated dimension.
        :param dimension: The dimension to get the limits for, type SinogramDimensions.
        :param segment_number: The segment number to get the limits for. Only _truly_ required for axial position.
        :return: A tuple containing the minimum and maximum value of the dimension (min, max).
        """
        if self.projdata is None:
            return 0, 0

        if dimension == ProjDataDims.SEGMENT_NUM:
            return self.projdata.get_min_segment_num(), \
                   self.projdata.get_max_segment_num()
        elif dimension == ProjDataDims.AXIAL_POS:
            return self.projdata.get_min_axial_pos_num(segment_number), \
                   self.projdata.get_max_axial_pos_num(segment_number)
        elif dimension == ProjDataDims.VIEW_NUMBER:
            return self.projdata.get_min_view_num(), \
                   self.projdata.get_max_view_num()
        elif dimension == ProjDataDims.TANGENTIAL_POS:
            return self.projdata.get_min_tangential_pos_num(), \
                   self.projdata.get_max_tangential_pos_num()
        elif dimension == ProjDataDims.TIMING_POS:
            return self.projdata.get_min_tof_pos_num(), \
                   self.projdata.get_max_tof_pos_num()
        else:
            raise ValueError("Unknown sinogram dimension: " + str(dimension))

    def get_num_indices(self, dimension: ProjDataDims) -> int:
        """Returns the number of indices in the given dimension."""
        limits = self.get_limits(dimension, self.get_current_segment_num())
        return limits[1] - limits[0] + 1

    def get_current_segment_num(self) -> int:
        """Returns the segment number of the current segment data."""
        if self.segment_data is not None:
            return self.segment_data.get_segment_num()
        return 0
