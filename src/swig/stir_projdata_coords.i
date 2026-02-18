/*
    Copyright (C) 2023, 2024 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \brief Interface file for SWIG: stir::Bin, stir:DetectionPosition etc

  \author Kris Thielemans
*/

%shared_ptr(stir::DetectionPosition<unsigned int>);
%shared_ptr(stir::DetectionPositionPair<unsigned int>);

%attributeref(stir::DetectionPosition<unsigned int>, unsigned int, tangential_coord);
%attributeref(stir::DetectionPosition<unsigned int>, unsigned int, axial_coord);
%attributeref(stir::DetectionPosition<unsigned int>, unsigned int, radial_coord);
%include "stir/DetectionPosition.h"
ADD_REPR(stir::DetectionPosition, %arg(*$self))
%template(DetectionPosition) stir::DetectionPosition<unsigned int>;

%attributeref(stir::DetectionPositionPair<unsigned int>, int, timing_pos);
%attributeref(stir::DetectionPositionPair<unsigned int>, stir::DetectionPosition<unsigned int>, pos1);
%attributeref(stir::DetectionPositionPair<unsigned int>, stir::DetectionPosition<unsigned int>, pos2);
%include "stir/DetectionPositionPair.h"
ADD_REPR(stir::DetectionPositionPair, %arg(*$self))
%template(DetectionPositionPair) stir::DetectionPositionPair<unsigned int>;

%attributeref(stir::SegmentIndices, int, segment_num);
%attributeref(stir::SegmentIndices, int, timing_pos_num);
%attributeref(stir::ViewgramIndices, int, view_num);
%attributeref(stir::SinogramIndices, int, axial_pos_num);
%attributeref(stir::Bin, int, axial_pos_num);
%attributeref(stir::Bin, int, tangential_pos_num);
%attributeref(stir::Bin, int, timing_pos_num);
%attributeref(stir::Bin, int, time_frame_num);
%attribute(stir::Bin, float, bin_value, get_bin_value, set_bin_value);
%include "stir/SegmentIndices.h"
%include "stir/ViewgramIndices.h"
%include "stir/SinogramIndices.h"
%include "stir/Bin.h"
ADD_REPR(stir::Bin, %arg(*$self))
