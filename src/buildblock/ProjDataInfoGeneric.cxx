/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2009-10-18 Hammersmith Imanet Ltd
    Copyright (C) 2011, Kris Thielemans
    Copyright (C) 2017 ETH Zurich, Institute of Particle Physics and Astrophysics
    Copyright (C) 2013, 2018, 2021, University College London

    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/

/*!

  \file
  \ingroup projdata

  \brief Non-inline implementations of stir::ProjDataInfoGeneric

  \author Kris Thielemans
  \author Sanida Mustafovic
  \author PARAPET project
  \author Parisa Khateri
  \author Michael Roethlisberger
*/


#include "stir/ProjDataInfoGeneric.h"
#include "stir/LORCoordinates.h"
#include <algorithm>
#ifdef BOOST_NO_STRINGSTREAM
#include <strstream.h>
#else
#include <sstream>
#endif

#include "stir/round.h"
#include <math.h>

#ifndef STIR_NO_NAMESPACES
using std::min_element;
using std::max_element;
using std::min;
using std::max;
using std::swap;
using std::endl;
#endif

START_NAMESPACE_STIR

ProjDataInfoGeneric::
ProjDataInfoGeneric()
{}


ProjDataInfoGeneric::
ProjDataInfoGeneric(const shared_ptr<Scanner>& scanner_ptr,
                        const VectorWithOffset<int>& num_axial_pos_per_segment,
                        const VectorWithOffset<int>& min_ring_diff_v,
                        const VectorWithOffset<int>& max_ring_diff_v,
                        const int num_views,const int num_tangential_poss)
  :ProjDataInfoCylindrical(scanner_ptr,num_axial_pos_per_segment,
                           min_ring_diff_v, max_ring_diff_v, num_views,num_tangential_poss)
{
}

#if 0
/*! Default implementation checks common variables. Needs to be overloaded.
 */
bool
ProjDataInfoGeneric::
blindly_equals(const root_type * const that) const
{
  if (!base_type::blindly_equals(that))
    return false;

  const self_type& proj_data_info = static_cast<const self_type&>(*that);
  return
    true;
}
#endif

void
ProjDataInfoGeneric::
set_num_views(const int new_num_views)
{
  if (new_num_views != get_num_views())
    error("ProjDataInfoGeneric::set_num_views not supported");
}
#if 0 // TODOBLOCK
void
ProjDataInfoGeneric::
set_ring_spacing(float ring_spacing_v)
{
  ring_diff_arrays_computed = false;
  ring_spacing = ring_spacing_v;
}
#endif

//! warning Find lor from cartesian coordinates of detector pair
void
ProjDataInfoGeneric::
get_LOR(LORInAxialAndNoArcCorrSinogramCoordinates<float>& lor,
	const Bin& bin) const
{
	CartesianCoordinate3D<float> _p1;
	CartesianCoordinate3D<float> _p2;
	find_cartesian_coordinates_of_detection(_p1, _p2, bin);
    
    _p1.z()+=z_shift.z();
    _p2.z()+=z_shift.z();
    
	LORAs2Points<float> lor_as_2_points(_p1, _p2);
	const double R = sqrt(max(square(_p1.x())+square(_p1.y()), square(_p2.x())+square(_p2.y())));
    
    lor_as_2_points.change_representation(lor, R);
}

std::string
ProjDataInfoGeneric::parameter_info()  const
{

#ifdef BOOST_NO_STRINGSTREAM
  // dangerous for out-of-range, but 'old-style' ostrstream seems to need this
  char str[30000];
  ostrstream s(str, 30000);
#else
  std::ostringstream s;
#endif
  s << ProjDataInfo::parameter_info();
  // TODOBLOCK Cylindrical has the following which doesn't make sense for Generic, so repeat code
  //s << "Azimuthal angle increment (deg):   " << get_azimuthal_angle_sampling()*180/_PI << '\n';
  //s << "Azimuthal angle extent (deg):      " << fabs(get_azimuthal_angle_sampling())*get_num_views()*180/_PI << '\n';

  s << "ring differences per segment: \n";
  for (int segment_num=get_min_segment_num(); segment_num<=get_max_segment_num(); ++segment_num)
  {
    s << '(' << get_min_ring_difference(segment_num)  << ',' << get_max_ring_difference(segment_num) <<')';
  }
  s << std::endl;
  return s.str();
}

END_NAMESPACE_STIR
