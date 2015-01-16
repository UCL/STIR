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
#include <math.h>

START_NAMESPACE_STIR

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
#pragma omp critical(PROJDATAINFOCYLINDRICALNOARCCORR_VIEWTANGPOS_TO_DETS)
  { 
    if (!uncompressed_view_tangpos_to_det1det2_initialised)
      initialise_uncompressed_view_tangpos_to_det1det2();
  }

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
#pragma omp critical(PROJDATAINFOCYLINDRICALNOARCCORR_DETS_TO_VIEWTANGPOS)
  {
    if (!det1det2_to_uncompressed_view_tangpos_initialised)
      initialise_det1det2_to_uncompressed_view_tangpos();

  }

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
		     const int det_num2, const int ring_num2) const
{  
  if (get_view_tangential_pos_num_for_det_num_pair(bin.view_num(), bin.tangential_pos_num(), det_num1, det_num2))
    return get_segment_axial_pos_num_for_ring_pair(bin.segment_num(), bin.axial_pos_num(), ring_num1, ring_num2);
  else
    return get_segment_axial_pos_num_for_ring_pair(bin.segment_num(), bin.axial_pos_num(), ring_num2, ring_num1);
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
                         dp.pos2().axial_coord());
}
void
ProjDataInfoCylindricalNoArcCorr::
get_det_pair_for_bin(
		     int& det_num1, int& ring_num1,
		     int& det_num2, int& ring_num2,
		     const Bin& bin) const
{
  get_det_num_pair_for_view_tangential_pos_num(det_num1, det_num2, bin.view_num(), bin.tangential_pos_num());
  get_ring_pair_for_segment_axial_pos_num( ring_num1, ring_num2, bin.segment_num(), bin.axial_pos_num());
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

#else

  get_det_pair_for_bin(dp.pos1().tangential_coord(),
                       dp.pos1().axial_coord(),
		       dp.pos2().tangential_coord(),
                       dp.pos2().axial_coord(),
                       bin);
#endif
}

END_NAMESPACE_STIR

