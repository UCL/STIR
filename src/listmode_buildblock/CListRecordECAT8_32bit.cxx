/*
    Copyright (C) 2013, 2023 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup listmode
  \brief Implementation of classes stir::ecat::CListEventECAT8_32bit and stir::ecat::CListRecordECAT8_32bit 
  for listmode events for the ECAT8 32bit listmode file format.
    
  \author Kris Thielemans
*/

#include "stir/listmode/CListRecordECAT8_32bit.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/Bin.h"
#include "stir/error.h"
#include "stir/Succeeded.h"
#include "stir/IO/stir_ecat_common.h"
#include <algorithm>
#include <iostream>
START_NAMESPACE_STIR
namespace ecat {

CListEventECAT8_32bit::
CListEventECAT8_32bit(const shared_ptr<const ProjDataInfo>& proj_data_info_sptr) :
  CListEventCylindricalScannerWithDiscreteDetectors(proj_data_info_sptr)
{
  const ProjDataInfoCylindricalNoArcCorr * const proj_data_info_ptr =
    dynamic_cast<const ProjDataInfoCylindricalNoArcCorr * const>(proj_data_info_sptr.get());
  if (proj_data_info_ptr == 0)
    error("CListEventECAT8_32bit can only be initialised with cylindrical projection data without arc-correction");
 if (proj_data_info_ptr->get_max_ring_difference(0) != proj_data_info_ptr->get_min_ring_difference(0))
   error("CListEventECAT8_32bit can only handle axial compression==1");

 this->segment_sequence = ecat::find_segment_sequence(*proj_data_info_ptr);
 this->sizes.resize(this->segment_sequence.size());
 for (std::size_t s=0U; s < this->segment_sequence.size(); ++s)
   this->sizes[s]=proj_data_info_ptr->get_num_axial_poss(segment_sequence[s]);
 this->timing_poss_sequence = ecat::find_timing_poss_sequence(*proj_data_info_sptr);
}

void
CListEventECAT8_32bit::
get_detection_position(DetectionPositionPair<>& det_pos) const
{
  /* data is organised by segment, axial coordinate, view, tangential */
  const int num_tangential_poss = this->get_uncompressed_proj_data_info_sptr()->get_num_tangential_poss();
  const int num_views = this->get_uncompressed_proj_data_info_sptr()->get_num_views();
  const int num_non_tof_sinograms = this->get_uncompressed_proj_data_info_sptr()->get_num_non_tof_sinograms();

  const int tang_pos_num = this->data.offset % num_tangential_poss;//(this->num_sinograms * this-> num_views);
  int rest = this->data.offset / num_tangential_poss;
  const int view_num = rest % num_views;
  rest = rest / num_views;
  int z = rest % num_non_tof_sinograms;
  const int Siemens_TOF_bin = rest / num_non_tof_sinograms;
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
  const int timing_pos_num = this->timing_poss_sequence[Siemens_TOF_bin];
  // this is actually a compressed bin for many Siemens scanners. would have to go to det_pos somehow, or overload get_bin
  const Bin uncompressed_bin(segment_num, view_num, axial_pos_num, tang_pos_num - (num_tangential_poss/2), timing_pos_num);
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
