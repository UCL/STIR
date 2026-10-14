/*
    Copyright (C) 2019 University of Hull
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup listmode SimSET
  \brief Implementation of classes stir::ecat::CListEventSimSET and stir::ecat::CListRecordSimsET
  for listmode events for the SimSET as listmode file format.

  \author Nikos Efthimiou
*/

#include "stir/listmode/CListRecordSimSET.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoBlocksOnCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoGenericNoArcCorr.h"
#include "stir/format.h"
#include "stir/LORCoordinates.h"

START_NAMESPACE_STIR

Succeeded
CListRecordSimSET::init_from_data(const PHG_DetectedPhoton& detectedPhotonBlue,
                                  const PHG_DetectedPhoton& detectedPhotonPink,
                                  const float weight,
                                  const bool is_prompt)
{

  double tof_difference = 1.E3F * (detectedPhotonPink.time_since_creation - detectedPhotonBlue.time_since_creation);

  CartesianCoordinate3D<float> coord_1(detectedPhotonBlue.location.z_position - half_ring_spacing,
                                       detectedPhotonBlue.location.y_position,
                                       detectedPhotonBlue.location.x_position);

  CartesianCoordinate3D<float> coord_2(detectedPhotonPink.location.z_position - half_ring_spacing,
                                       detectedPhotonPink.location.y_position,
                                       detectedPhotonPink.location.x_position);

  const LORAs2Points<float> input_lor(coord_1, coord_2);
  LORAs2Points<float> intersected_lor;

  if (input_lor.get_intersections_with_cylinder(intersected_lor, radius) == Succeeded::no)
    return Succeeded::no;

  this->event_data->set_lor(intersected_lor, tof_difference, weight);
  this->event_data->set_prompt(is_prompt);

  const auto time_in_millisecs = static_cast<unsigned long>(detectedPhotonBlue.time_since_creation / 1.E6);

  this->time_data.set_time_in_millisecs(time_in_millisecs);

  return Succeeded::yes;
}

std::unique_ptr<CListEventScannerWithDiscreteDetectorsBase>
CListRecordSimSET::make_event_data(shared_ptr<const ProjDataInfo> proj_data_info_sptr)
{
  // construct event of type of current ProjDataInfo
  // Note: currently cumbersome due to change ProjDataInfo hierarchy.
  // The following is safe...
  // See https://github.com/UCL/STIR/commit/79bd05694091f7b08fb0237cb34bdbeedb256a45
  if ((proj_data_info_sptr->get_scanner_ptr()->get_scanner_geometry() == "Cylindrical")
      && (dynamic_cast<const ProjDataInfoCylindricalNoArcCorr*>(proj_data_info_sptr.get()) != nullptr))
    {
      return std::make_unique<CListEventSimSET<ProjDataInfoCylindricalNoArcCorr>>(proj_data_info_sptr);
    }

  if ((proj_data_info_sptr->get_scanner_ptr()->get_scanner_geometry() == "BlocksOnCylindrical")
      && (dynamic_cast<const ProjDataInfoBlocksOnCylindricalNoArcCorr*>(proj_data_info_sptr.get()) != nullptr))
    {
      return std::make_unique<CListEventSimSET<ProjDataInfoBlocksOnCylindricalNoArcCorr>>(proj_data_info_sptr);
    }

  if ((proj_data_info_sptr->get_scanner_ptr()->get_scanner_geometry() == "Generic")
      && (dynamic_cast<const ProjDataInfoGenericNoArcCorr*>(proj_data_info_sptr.get()) != nullptr))
    {
      return std::make_unique<CListEventSimSET<ProjDataInfoGenericNoArcCorr>>(proj_data_info_sptr);
    }

  error("Unsupported ProjDataInfo type in CListRecordPETSIRD::make_event_data");
  return nullptr;
}

END_NAMESPACE_STIR
