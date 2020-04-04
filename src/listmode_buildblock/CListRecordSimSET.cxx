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

START_NAMESPACE_STIR


CListEventSimSET::
CListEventSimSET(const shared_ptr<ProjDataInfo> &proj_data_info) :
    CListEventCylindricalScannerWithDiscreteDetectors(proj_data_info)
{

}

void CListEventSimSET::get_detection_position(DetectionPositionPair<>& _det_pos) const
{
    DetectionPosition<> det1(this->det1, this->ring1, 0);
    DetectionPosition<> det2(this->det2, this->ring2, 0);

    _det_pos.pos1() = det1;
    _det_pos.pos2() = det2;
    _det_pos.timing_pos() = this->get_uncompressed_proj_data_info_sptr()->get_unmashed_tof_bin(tofDifference);
}

void CListEventSimSET::set_detection_position(const DetectionPositionPair<>&)
{
    error("Cannot set events in a ROOT file!");
}

void CListEventSimSET::init_from_data(const PHG_DetectedPhoton* _blue,
                                      const PHG_DetectedPhoton* _pink,
                                      const float _weight,
                                      const float _tofDifference)
{

    CartesianCoordinate3D<float> blue_coord(_blue->location.z_position,
                                             _blue->location.y_position,
                                             _blue->location.x_position);

    CartesianCoordinate3D<float> pink_coord(_pink->location.z_position,
                                             _pink->location.y_position,
                                             _pink->location.x_position);

    weight = _weight;
    tofDifference = _tofDifference;

    this->get_uncompressed_proj_data_info_sptr()->
            find_scanner_coordinates_given_cartesian_coordinates(det2, det1,
                                                                 ring2, ring1,
                                                                 blue_coord,
                                                                 pink_coord);

}

END_NAMESPACE_STIR
