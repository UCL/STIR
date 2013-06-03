//
// $Id: CListRecordECAT8_32bit.cxx,v 1.5 2011-06-24 15:36:55 kris Exp $
//
/*
    Copyright (C) 1998-2011, Hammersmith Imanet Ltd
    Copyright (C) 2013 University College London
*/
/*!
  \file
  \ingroup listmode
  \brief Implementation of classes CListEventECAT8_32bit and CListRecordECAT8_32bit 
  for listmode events for the ECAT8 32bit listmode file format.
    
  \author Kris Thielemans
      
  $Date: 2011-06-24 15:36:55 $
  $Revision: 1.5 $
*/

#include "UCL/listmode/CListRecordECAT8_32bit.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/Bin.h"
#include "stir/error.h"
#include "stir/Succeeded.h"

#include <algorithm>
#include <iostream>
START_NAMESPACE_STIR
namespace UCL {


void
CListEventECAT8_32bit::
get_detection_position(DetectionPositionPair<>& det_pos) const
{
  /* data is organised by segment, axial coordinate, view, tangential */
  const int num_tangential_poss = this->scanner_sptr->get_default_num_arccorrected_bins();
  const int num_views = this->scanner_sptr->get_num_detectors_per_ring()/2;

  const int tang_pos_num = this->data.offset % num_tangential_poss;//(this->num_sinograms * this-> num_views);
  int rest = this->data.offset / num_tangential_poss;
  const int view_num = rest % num_views;
  int z = rest / num_views;
  // TODO
  int axial_pos_num = 0;
  int segment_num = 0;
    vector<int> segment_sequence(121);
    vector<int> sizes(121);
    segment_sequence[0]=0;
    sizes[0]=64;
    for (int ringdiff=1; ringdiff<=60; ++ringdiff)
    {
        segment_sequence[2*ringdiff-1]=ringdiff;
        segment_sequence[2*ringdiff]=-ringdiff;
        sizes[2*ringdiff-1]=64-ringdiff;
        sizes[2*ringdiff]=64-ringdiff;
    }
    for (int i=0; i<segment_sequence.size();++i)
    {
        if (z< sizes[i])
        {
            axial_pos_num = z;
            segment_num = segment_sequence[i];
            break;
        }
        else
        {
            z -= sizes[i];
        }
    }
/*
 //   for segment 0
  if (z<this->get_uncompressed_proj_data_info_sptr()->get_num_axial_poss(0))
  {
      axial_pos_num = z;
      segment_num = 0;
  }
    else
    {
       segment_num = 1;
       axial_pos_num=0;
    }
*/
//    if (segment_num==0 && axial_pos_num==0 && (this->data.offset == 66753-1 || tang_pos_num == 16 || view_num == 193) )
//        std::cerr << "offset " << this->data.offset << "t " << tang_pos_num << " v" << view_num << std::endl;
  Bin uncompressed_bin(segment_num, view_num, axial_pos_num,tang_pos_num - (num_tangential_poss/2));
  this->get_uncompressed_proj_data_info_sptr()->get_det_pos_pair_for_bin(det_pos,uncompressed_bin);  
}

void
CListEventECAT8_32bit::
set_detection_position(const DetectionPositionPair<>&)
{
  error("cannot set events yet");
}



} // namespace UCL 

END_NAMESPACE_STIR
