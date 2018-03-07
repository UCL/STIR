//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
    Copyright (C) 2018, University College London
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
  \ingroup projdata

  \brief Implementation of non-inline functions of class 
  stir::ProjDataInfoCylindricalArcCorr

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project


*/

#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/Bin.h"
#include "stir/round.h"
#include "stir/LORCoordinates.h"
#ifdef BOOST_NO_STRINGSTREAM
#include <strstream.h>
#else
#include <sstream>
#endif

#ifndef STIR_NO_NAMESPACES
using std::endl;
using std::ends;
using std::string;
#endif


START_NAMESPACE_STIR
ProjDataInfoCylindricalArcCorr:: ProjDataInfoCylindricalArcCorr()
{}

ProjDataInfoCylindricalArcCorr:: ProjDataInfoCylindricalArcCorr(const shared_ptr<Scanner> scanner_ptr,float bin_size_v,								
								const  VectorWithOffset<int>& num_axial_pos_per_segment,
								const  VectorWithOffset<int>& min_ring_diff_v, 
								const  VectorWithOffset<int>& max_ring_diff_v,
                                const int num_views,const int num_tangential_poss,
                                const int tof_mash_factor)
								:ProjDataInfoCylindrical(scanner_ptr,
								num_axial_pos_per_segment,
								min_ring_diff_v, max_ring_diff_v,
								num_views, num_tangential_poss),
								bin_size(bin_size_v)								
								
{
	if (scanner_ptr->is_tof_ready())
        set_tof_mash_factor(tof_mash_factor);
}


void
ProjDataInfoCylindricalArcCorr::set_tangential_sampling(const float new_tangential_sampling)
{bin_size = new_tangential_sampling;}



ProjDataInfo*
ProjDataInfoCylindricalArcCorr::clone() const
{
  return static_cast<ProjDataInfo*>(new ProjDataInfoCylindricalArcCorr(*this));
}


bool
ProjDataInfoCylindricalArcCorr::
operator==(const self_type& that) const
{
  if (!base_type::blindly_equals(&that))
    return false;
  return
    fabs(this->bin_size - that.bin_size) < 0.05F;
}

bool
ProjDataInfoCylindricalArcCorr::
blindly_equals(const root_type * const that_ptr) const
{
  assert(dynamic_cast<const self_type * const>(that_ptr) != 0);
  return
    this->operator==(static_cast<const self_type&>(*that_ptr));
}

string
ProjDataInfoCylindricalArcCorr::parameter_info()  const
{

#ifdef BOOST_NO_STRINGSTREAM
  // dangerous for out-of-range, but 'old-style' ostrstream seems to need this
  char str[50000];
  ostrstream s(str, 50000);
#else
  std::ostringstream s;
#endif  
  s << "ProjDataInfoCylindricalArcCorr := \n";
  s << ProjDataInfoCylindrical::parameter_info();
  s << "tangential sampling := " << get_tangential_sampling() << endl;
  s << "End :=\n";
  return s.str();
}


Bin
ProjDataInfoCylindricalArcCorr::
get_bin(const LOR<float>& lor,const double delta_time) const

{
  if (delta_time != 0)
    {
	  error("TODO NO TOF YET");
    }

  Bin bin;
  LORInAxialAndSinogramCoordinates<float> lor_coords;
  if (lor.change_representation(lor_coords, get_ring_radius()) == Succeeded::no)
    {
      bin.set_bin_value(-1);
      return bin;
    }

  // first find view 
  // unfortunately, phi ranges from [0,Pi[, but the rounding can
  // map this to a view which corresponds to Pi anyway.
  bin.view_num() = round(lor_coords.phi() / get_azimuthal_angle_sampling());
  assert(bin.view_num()>=0);
  assert(bin.view_num()<=get_num_views());
  const bool swap_direction =
    bin.view_num() > get_max_view_num();
  if (swap_direction)
    bin.view_num()-=get_num_views();

  bin.tangential_pos_num() = round(lor_coords.s() / get_tangential_sampling());
  if (swap_direction)
    bin.tangential_pos_num() *= -1;

  if (bin.tangential_pos_num() < get_min_tangential_pos_num() ||
      bin.tangential_pos_num() > get_max_tangential_pos_num())
    {
      bin.set_bin_value(-1);
      return bin;
    }

#if 0
  const int num_rings = 
    get_scanner_ptr()->get_num_rings();
  // TODO WARNING LOR coordinates are w.r.t. centre of scanner, but the rings are numbered with the first ring at 0
  int ring1, ring2;
  if (!swap_direction)
    {
      ring1 = round(lor_coords.z1()/get_ring_spacing() + (num_rings-1)/2.F);
      ring2 = round(lor_coords.z2()/get_ring_spacing() + (num_rings-1)/2.F);
    }
  else
    {
      ring2 = round(lor_coords.z1()/get_ring_spacing() + (num_rings-1)/2.F);
      ring1 = round(lor_coords.z2()/get_ring_spacing() + (num_rings-1)/2.F);
    }

  if (!(ring1 >=0 && ring1<get_scanner_ptr()->get_num_rings() &&
	ring2 >=0 && ring2<get_scanner_ptr()->get_num_rings() &&
	get_segment_axial_pos_num_for_ring_pair(bin.segment_num(),
						bin.axial_pos_num(),
						ring1,
						ring2) == Succeeded::yes)
      )
    {
      bin.set_bin_value(-1);
      return bin;
    }
#else
  // find nearest segment
  {
    const float delta =
      (swap_direction 
       ? lor_coords.z1()-lor_coords.z2()
       : lor_coords.z2()-lor_coords.z1()
       )/get_ring_spacing();
    // check if out of acquired range
    // note the +1 or -1, which takes the size of the rings into account
    if (delta>get_max_ring_difference(get_max_segment_num())+1 ||
	delta<get_min_ring_difference(get_min_segment_num())-1)
      {
	bin.set_bin_value(-1);
	return bin;
      } 
    if (delta>=0)
      {
	for (bin.segment_num()=0; bin.segment_num()<get_max_segment_num(); ++bin.segment_num())
	  {
	    if (delta < get_max_ring_difference(bin.segment_num())+.5)
	      break;
	  }
      }
    else
      {
	// delta<0
	for (bin.segment_num()=0; bin.segment_num()>get_min_segment_num(); --bin.segment_num())
	  {
	    if (delta > get_min_ring_difference(bin.segment_num())-.5)
	      break;
	  }
      }
  }
  // now find nearest axial position
  {
    const float m = (lor_coords.z2()+lor_coords.z1())/2;
#if 0
    // this uses private member of ProjDataInfoCylindrical
    // enable when moved
    initialise_ring_diff_arrays_if_not_done_yet();

#ifndef NDEBUG
    bin.axial_pos_num()=0;
    assert(get_m(bin)==- m_offset[bin.segment_num()]);
#endif
    bin.axial_pos_num() =
      round((m + m_offset[bin.segment_num()])/
	    get_axial_sampling(bin.segment_num()));
#else
    bin.axial_pos_num()=0;
    bin.axial_pos_num() =
      round((m - get_m(bin))/
	    get_axial_sampling(bin.segment_num()));
#endif
    if (bin.axial_pos_num() < get_min_axial_pos_num(bin.segment_num()) ||
	bin.axial_pos_num() > get_max_axial_pos_num(bin.segment_num()))
      {
	bin.set_bin_value(-1);
	return bin;
      }
  }
#endif

  bin.set_bin_value(1);
  return bin;
}
END_NAMESPACE_STIR

