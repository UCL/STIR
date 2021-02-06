//
//
/*!

  \file
  \ingroup projdata

  \brief Implementation of inline functions of class 
  ProjDataInfoCylindricalNoArcCorr

  \author Kris Thielemans

*/
/*
    Copyright (C) 2000- 2005, Hammersmith Imanet Ltd
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

#include "stir/Bin.h"
#include "stir/Succeeded.h"
#include "stir/round.h"
#include <math.h>

START_NAMESPACE_STIR

void 
ProjDataInfoCylindricalNoArcCorr::
initialise_uncompressed_view_tangpos_to_det1det2_if_not_done_yet() const
{
  // for efficiency reasons, use "Double-Checked-Locking(DCL) pattern" with OpenMP atomic operation
  // OpenMP v3.1 or later required
  // thanks to yohjp: http://stackoverflow.com/questions/27975737/how-to-handle-cached-data-structures-with-multi-threading-e-g-openmp
#if defined(STIR_OPENMP) &&  _OPENMP >=201012
  bool initialised;
#pragma omp atomic read
  initialised = uncompressed_view_tangpos_to_det1det2_initialised;

  if (!initialised)
#endif
    {
#if defined(STIR_OPENMP)
#pragma omp critical(PROJDATAINFOCYLINDRICALNOARCCORR_VIEWTANGPOS_TO_DETS)
#endif
          { 
            if (!uncompressed_view_tangpos_to_det1det2_initialised)
              initialise_uncompressed_view_tangpos_to_det1det2();
          }
    }
}

void 
ProjDataInfoCylindricalNoArcCorr::
initialise_det1det2_to_uncompressed_view_tangpos_if_not_done_yet() const
{
  // as above
#if defined(STIR_OPENMP) &&  _OPENMP >=201012
  bool initialised;
#pragma omp atomic read
  initialised = det1det2_to_uncompressed_view_tangpos_initialised;

  if (!initialised)
#endif
    {
#if defined(STIR_OPENMP)
#pragma omp critical(PROJDATAINFOCYLINDRICALNOARCCORR_DETS_TO_VIEWTANGPOS)
#endif
          { 
            if (!det1det2_to_uncompressed_view_tangpos_initialised)
              initialise_det1det2_to_uncompressed_view_tangpos();
          }
    }
}

float
ProjDataInfoCylindricalNoArcCorr::
get_s(const Bin& bin) const
{
  return ring_radius * sin(bin.tangential_pos_num()*angular_increment);
}

float
ProjDataInfoCylindricalNoArcCorr::
get_angular_increment() const
{
  return angular_increment;
}

void
ProjDataInfoCylindricalNoArcCorr::
get_det_num_pair_for_view_tangential_pos_num(
					     int& det1_num,
					     int& det2_num,
					     const int view_num,
					     const int tang_pos_num) const
{
  assert(get_view_mashing_factor() == 1);
  this->initialise_uncompressed_view_tangpos_to_det1det2_if_not_done_yet();

  det1_num = uncompressed_view_tangpos_to_det1det2[view_num][tang_pos_num].det1_num;
  det2_num = uncompressed_view_tangpos_to_det1det2[view_num][tang_pos_num].det2_num;
}


bool 
ProjDataInfoCylindricalNoArcCorr::
get_view_tangential_pos_num_for_det_num_pair(int& view_num,
					     int& tang_pos_num,
					     const int det1_num,
					     const int det2_num) const
{
  assert(det1_num!=det2_num);
  this->initialise_det1det2_to_uncompressed_view_tangpos_if_not_done_yet();

  view_num = 
    det1det2_to_uncompressed_view_tangpos[det1_num][det2_num].view_num/get_view_mashing_factor();
  tang_pos_num = 
    det1det2_to_uncompressed_view_tangpos[det1_num][det2_num].tang_pos_num;
  return
    det1det2_to_uncompressed_view_tangpos[det1_num][det2_num].swap_detectors;
}


Succeeded 
ProjDataInfoCylindricalNoArcCorr::
get_bin_for_det_pair(Bin& bin,
		     const int det_num1, const int ring_num1,
		     const int det_num2, const int ring_num2,
			 const int timing_pos_num) const
{  
  if (get_view_tangential_pos_num_for_det_num_pair(bin.view_num(), bin.tangential_pos_num(), det_num1, det_num2))
  {
	bin.timing_pos_num() = timing_pos_num;
	return get_segment_axial_pos_num_for_ring_pair(bin.segment_num(), bin.axial_pos_num(), ring_num1, ring_num2);
  }
  else
  {
	bin.timing_pos_num() = -timing_pos_num;
    return get_segment_axial_pos_num_for_ring_pair(bin.segment_num(), bin.axial_pos_num(), ring_num2, ring_num1);
  }
}

Succeeded 
ProjDataInfoCylindricalNoArcCorr::
get_bin_for_det_pos_pair(Bin& bin,
                         const DetectionPositionPair<>& dp) const
{
  return
    get_bin_for_det_pair(bin,
                         dp.pos1().tangential_coord(),
                         dp.pos1().axial_coord(),
                         dp.pos2().tangential_coord(),
                         dp.pos2().axial_coord(),
                         this->get_tof_mash_factor()==0
                           ? 0 // use timing_pos==0 in the nonTOF case
                           : stir::round((float)dp.timing_pos()/this->get_tof_mash_factor()));
}
void
ProjDataInfoCylindricalNoArcCorr::
get_det_pair_for_bin(
             int& det_num1, int& ring_num1,
             int& det_num2, int& ring_num2,
             const Bin& bin) const
{
  //if (bin.timing_pos_num()>=0)
 // {
    get_det_num_pair_for_view_tangential_pos_num(det_num1, det_num2, bin.view_num(), bin.tangential_pos_num());
    get_ring_pair_for_segment_axial_pos_num( ring_num1, ring_num2, bin.segment_num(), bin.axial_pos_num());
  //}
  //else
  //{
  //  get_det_num_pair_for_view_tangential_pos_num(det_num2, det_num1, bin.view_num(), bin.tangential_pos_num());
  //  get_ring_pair_for_segment_axial_pos_num( ring_num2, ring_num1, bin.segment_num(), bin.axial_pos_num());
 // }
}

void
ProjDataInfoCylindricalNoArcCorr::
get_det_pos_pair_for_bin(
             DetectionPositionPair<>& dp,
             const Bin& bin) const
{
  //lousy work around because types don't match TODO remove!
#if 1
  int t1=dp.pos1().tangential_coord(),
    a1=dp.pos1().axial_coord(),
    t2=dp.pos2().tangential_coord(),
    a2=dp.pos2().axial_coord();
  get_det_pair_for_bin(t1, a1, t2, a2, bin);
  dp.pos1().tangential_coord()=t1;
  dp.pos1().axial_coord()=a1;
  dp.pos2().tangential_coord()=t2;
  dp.pos2().axial_coord()=a2;
  dp.timing_pos() = std::abs(bin.timing_pos_num())*this->get_tof_mash_factor();

#else

  get_det_pair_for_bin(dp.pos1().tangential_coord(),
                       dp.pos1().axial_coord(),
               dp.pos2().tangential_coord(),
                       dp.pos2().axial_coord(),
                       bin);
#endif
}

END_NAMESPACE_STIR

