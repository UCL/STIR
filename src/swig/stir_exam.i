/*
    Copyright (C) 2022 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \brief Interface file for SWIG: stir::ExamData, stir::ExamInfo and components

  \author Kris Thielemans

*/
%shared_ptr(stir::TimeFrameDefinitions);
%shared_ptr(stir::ImagingModality);
%shared_ptr(stir::PatientPosition);
%shared_ptr(stir::RadionuclideDB);
%shared_ptr(stir::Radionuclide);
%shared_ptr(stir::ExamInfo);
%shared_ptr(stir::ExamData);

%include "stir/TimeFrameDefinitions.h"
%include "stir/ImagingModality.h"
%include "stir/PatientPosition.h"
%include "stir/Radionuclide.h"
%include "stir/RadionuclideDB.h"
%include "stir/ExamInfo.h"
%include "stir/ExamData.h"

// add display functions
ADD_REPR(stir::PatientPosition, %arg($self->get_position_as_string()));
ADD_REPR(stir::ImagingModality, %arg($self->get_name()));
ADD_REPR_PARAMETER_INFO(stir::Radionuclide);
ADD_REPR_PARAMETER_INFO(stir::ExamInfo);

