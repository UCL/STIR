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
  \brief Implementation of classes stir::ecat::CListEventECAT8_32bit and stir::ecat::CListRecordECAT8_32bit 
  for listmode events for the ECAT8 32bit listmode file format.
    
  \author Kris Thielemans
*/

#include "stir/listmode/CListRecordPENN.h"


START_NAMESPACE_STIR

CListEventPENN::
CListEventPENN(const shared_ptr<Scanner> &scanner_sptr) :
  CListEventCylindricalScannerWithDiscreteDetectors(scanner_sptr)
{
    quarter_of_detectors = static_cast<int>(scanner_sptr->get_num_detectors_per_ring()/4.f)-4;
}

void
CListEventPENN::
get_detection_position(DetectionPositionPair<>& det_pos) const
{
    DetectionPosition<> det1(d1, z1, 0);
    DetectionPosition<> det2(d2, z2, 0);
    det_pos.pos1() = det1;
    det_pos.pos2() = det2;

#ifdef STIR_TOF
    det_pos.timing_pos() = tof_bin;
#endif
}

void
CListEventPENN::
set_detection_position(const DetectionPositionPair<>&)
{
  error("cannot set events yet");
}

void
CListEventPENN::init_from_data(bool d,
                               int dt,
                               int xa, int xb,
                               int za, int zb,
                               int ea, int eb)
{
    delay = d;

    d1 = xa ;//+ quarter_of_detectors;
    d2 = xb ;//+ quarter_of_detectors;

    d1 = d1 + static_cast<int>((d1+16) / 32); // add gap
    d2 = d2 + static_cast<int>((d2+16) / 32); // add gap

    // get ring 1
    int r = za * 0.025f;
    // for each ring add 16 rings gap
    z1 = za + 16*(r + 1);
    // get ring 2
    r = zb * 0.025f;
    z2 = zb + 16*(r + 1);

//    orig_z1 = static_cast<unsigned short int>(za);
  //  orig_z2 = static_cast<unsigned short int>(zb);

    //energy1 = ea;// * 0.511f;
    //energy2 = eb;// * 0.511f;

    #ifdef STIR_TOF
    orig_tof_bin = static_cast<short int>(dt);
    tof_bin =orig_tof_bin;
    if (tof_bin < 0)
        tof_bin = 0;
    #endif

    if  (d1 < 0 )
        d1 = this->get_uncompressed_proj_data_info_sptr()->get_scanner_ptr()->get_num_detectors_per_ring() + d1;
    else if ( d1 >= this->get_uncompressed_proj_data_info_sptr()->get_scanner_ptr()->get_num_detectors_per_ring())
        d1 = d1 - this->get_uncompressed_proj_data_info_sptr()->get_scanner_ptr()->get_num_detectors_per_ring();

    if  (d2 < 0 )
        d2 = this->get_uncompressed_proj_data_info_sptr()->get_scanner_ptr()->get_num_detectors_per_ring() + d2;
    else if ( d2 >= this->get_uncompressed_proj_data_info_sptr()->get_scanner_ptr()->get_num_detectors_per_ring())
        d2 = d2 - this->get_uncompressed_proj_data_info_sptr()->get_scanner_ptr()->get_num_detectors_per_ring();


//      energy1 = stoi(results[8]);// * 0.511f;
//      energy2 = stoi(results[9]);// * 0.511f;

}

END_NAMESPACE_STIR
