//
// $Id: CListRecordECAT8_32bit.cxx,v 1.5 2011-06-24 15:36:55 kris Exp $
//
/*
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

CListEventECAT8_32bit::
CListEventECAT8_32bit() :
  CListEventCylindricalScannerWithDiscreteDetectors(shared_ptr<Scanner>(new Scanner(Scanner::Siemens_mMR)))
{
  const int num_rings = this->scanner_sptr->get_num_rings();
  // TODO remove hard-coding of sizes. depends on mMR (and acquisition?)
  const int max_ring_diff=60;
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



} // namespace UCL 

END_NAMESPACE_STIR
