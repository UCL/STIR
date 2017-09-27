/*
    Copyright (C) 2015-2016 University of Leeds
    Copyright (C) 2016 UCL
    Copyright (C) 2017, University of Hull
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
  \ingroup listmode
  \brief Implementation of classes stir::ecat::CListEventROOT and stir::ecat::CListRecordROOT
  for listmode events for the ROOT as listmode file format.

  \author Nikos Efthimiou
  \author Harry Tsoumpas
*/

#include "stir/listmode/CListRecordROOT.h"

START_NAMESPACE_STIR


CListEventROOT::
CListEventROOT(const shared_ptr<ProjDataInfo> &proj_data_info) :
    CListEventCylindricalScannerWithDiscreteDetectors(proj_data_info)
{
    quarter_of_detectors = static_cast<int>(proj_data_info->get_scanner_ptr()->get_num_detectors_per_ring()/4.f);
}

//!
//! \brief CListEventROOT::get_detection_position
//! \param det_pos
//! \author Nikos Efthimiou
//!
void CListEventROOT::get_detection_position(DetectionPositionPair<>& _det_pos) const
{

    DetectionPosition<> det1(this->det1, this->ring1, 0);
    DetectionPosition<> det2(this->det2, this->ring2, 0);

    _det_pos.pos1() = det1;
    _det_pos.pos2() = det2;
    _det_pos.timing_pos() = this->get_uncompressed_proj_data_info_sptr()->get_tof_bin(delta_time);
}

void CListEventROOT::set_detection_position(const DetectionPositionPair<>&)
{
    error("Cannot set events in a ROOT file!");
}

void CListEventROOT::init_from_data(const int& _ring1, const int& _ring2,
                                    const int& crystal1, const int& crystal2,
                                    const double& _delta_time)
{

    // STIR assumes that 0 is on y whill GATE on the x axis
    det1 = crystal1 + quarter_of_detectors;
    det2 = crystal2 + quarter_of_detectors;

    if  (det1 < 0 )
        det1 = this->uncompressed_proj_data_info_sptr->get_scanner_ptr()->get_num_detectors_per_ring() + det1;
    else if ( det1 >= this->uncompressed_proj_data_info_sptr->get_scanner_ptr()->get_num_detectors_per_ring())
        det1 = det1 - this->uncompressed_proj_data_info_sptr->get_scanner_ptr()->get_num_detectors_per_ring();

    if  (det2 < 0 )
        det2 = this->uncompressed_proj_data_info_sptr->get_scanner_ptr()->get_num_detectors_per_ring() + det2;
    else if ( det2 >= this->uncompressed_proj_data_info_sptr->get_scanner_ptr()->get_num_detectors_per_ring())
        det2 = det2 - this->uncompressed_proj_data_info_sptr->get_scanner_ptr()->get_num_detectors_per_ring();

    ring1 = _ring1;
    ring2 = _ring2;
    delta_time = _delta_time;
    swapped = false;
}

END_NAMESPACE_STIR
