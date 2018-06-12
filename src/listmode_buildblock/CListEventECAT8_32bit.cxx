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
  \brief Implementation of classes stir::ecat::CListEventECAT8_32bit
  for listmode events for the ECAT8 32bit listmode file format.
    
  \author Nikos Efthimiou
  \author Kris Thielemans
*/

#include "stir/listmode/CListEventECAT8_32bit.h"

START_NAMESPACE_STIR
namespace ecat {

CListEventECAT8_32bit::
CListEventECAT8_32bit(const shared_ptr<ProjDataInfo>& proj_data_info_sptr) :
  CListEventCylindricalScannerWithDiscreteDetectors(shared_ptr<Scanner>(new Scanner(*proj_data_info_sptr->get_scanner_ptr()))) 
{
  const ProjDataInfoCylindricalNoArcCorr * const proj_data_info_ptr =
    dynamic_cast<const ProjDataInfoCylindricalNoArcCorr * const>(proj_data_info_sptr.get());
  if (proj_data_info_ptr == 0)
    error("CListEventECAT8_32bit can only be initialised with cylindrical projection data without arc-correction");

  const int num_rings = this->scanner_sptr->get_num_rings();
  const int max_ring_diff=proj_data_info_ptr->get_max_ring_difference(proj_data_info_ptr->get_max_segment_num());
  if (proj_data_info_ptr->get_max_ring_difference(0) != proj_data_info_ptr->get_min_ring_difference(0))
    error("CListEventECAT8_32bit can only handle axial compression==1");
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

void
CListEventECAT8_32bit::
get_detection_position(DetectionPositionPair<>& det_pos) const
{
  /* data is organised by segment, axial coordinate, view, tangential */
  const int num_tangential_poss = this->scanner_sptr->get_default_num_arccorrected_bins();
  const int num_views = this->scanner_sptr->get_num_detectors_per_ring()/2;

  const int tang_pos_num = this->data.offset % num_tangential_poss;//(this->num_sinograms * this-> num_views);
  const int rest = this->data.offset / num_tangential_poss;
  const int view_num = rest % num_views;
  int z = rest / num_views;
  int axial_pos_num = 0;
  int segment_num = 0;
  for (std::size_t i=0; i<this->segment_sequence.size();++i)
    {
      if (z< this->sizes[i])
        {
          axial_pos_num = z;
          segment_num = this->segment_sequence[i];
          break;
        }
      else
        {
          z -= this->sizes[i];
        }
    }
  const Bin uncompressed_bin(segment_num, view_num, axial_pos_num,tang_pos_num - (num_tangential_poss/2));
  this->get_uncompressed_proj_data_info_sptr()->get_det_pos_pair_for_bin(det_pos,uncompressed_bin);  
}

void
CListEventECAT8_32bit::
set_detection_position(const DetectionPositionPair<>&)
{
  error("cannot set events yet");
}



} // namespace ecat

END_NAMESPACE_STIR
