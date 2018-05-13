/*
    Copyright (C) 2015-2016 University of Leeds
    Copyright (C) 2016 UCL
     Copyright (C) 2018 University of Hull
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

#include "stir/listmode/CListEventROOT.h"

START_NAMESPACE_STIR


CListEventROOT::
CListEventROOT(const shared_ptr<Scanner>& scanner_sptr) :
    CListEventCylindricalScannerWithDiscreteDetectors(scanner_sptr)
{
    quarter_of_detectors = static_cast<int>(scanner_sptr->get_num_detectors_per_ring()/4.f);
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
}

void CListEventROOT::set_detection_position(const DetectionPositionPair<>&)
{
    error("Cannot set events in a ROOT file!");
}

END_NAMESPACE_STIR

