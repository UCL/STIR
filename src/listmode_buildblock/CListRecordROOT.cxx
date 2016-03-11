/*
    Copyright (C) 2013 University College London
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
*/

#include "stir/listmode/CListRecordROOT.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/Bin.h"
#include "stir/error.h"
#include "stir/Succeeded.h"

#include <algorithm>
#include <iostream>
START_NAMESPACE_STIR


CListEventROOT::
CListEventROOT(const shared_ptr<ProjDataInfo>& proj_data_info_sptr) :
    CListEventCylindricalScannerWithDiscreteDetectors(shared_ptr<Scanner>(new Scanner(*proj_data_info_sptr->get_scanner_ptr())))
{

    const ProjDataInfoCylindricalNoArcCorr * const proj_data_info_ptr =
            dynamic_cast<const ProjDataInfoCylindricalNoArcCorr * const>(proj_data_info_sptr.get());

    if (proj_data_info_ptr == 0)
        error("CListEventDataROOT can only be initialised with cylindrical projection data without arc-correction");

    const int num_rings = this->scanner_sptr->get_num_rings();
    const int max_ring_diff=proj_data_info_ptr->get_max_ring_difference(proj_data_info_ptr->get_max_segment_num());

    if (proj_data_info_ptr->get_max_ring_difference(0) != proj_data_info_ptr->get_min_ring_difference(0))
        error("CListEventDataROOT can only handle axial compression==1");

    this->segment_sequence.resize(2*max_ring_diff+1);
    this->sizes.resize(2*max_ring_diff+1);
    this->segment_sequence[0]=0;
    this->sizes[0]=num_rings;

    for (int ringdiff=1; ringdiff<=max_ring_diff; ++ringdiff)
    {
        this->segment_sequence[2*ringdiff-1]=-ringdiff;
        this->segment_sequence[2*ringdiff]=ringdiff;
        this->sizes[2*ringdiff-1]=num_rings-ringdiff;
        this->sizes[2*ringdiff]=num_rings-ringdiff;
    }
}

//!
//! \brief CListEventROOT::get_detection_position
//! \param det_pos
//! \author Nikos Efthimiou
//!
Succeeded
CListEventROOT::
get_detection_position(DetectionPositionPair<>& det_pos) const
{

    DetectionPosition<> det1(this->data.det1, this->data.ring1, 0);
    DetectionPosition<> det2(this->data.det2, this->data.ring2, 0);

    det_pos.pos1() = det1;
    det_pos.pos2() = det2;

    return Succeeded::yes;
}

void
CListEventROOT::
set_detection_position(const DetectionPositionPair<>&)
{
    error("cannot set events yet");
}



END_NAMESPACE_STIR
