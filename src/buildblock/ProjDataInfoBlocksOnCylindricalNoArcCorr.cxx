
/*
    Copyright (C) 2000- 2007-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2011, Kris Thielemans
    Copyright (C) 2017, ETH Zurich, Institute of Particle Physics and Astrophysics
    Copyright (C) 2018, University College London
    Copyright (C) 2018, University of Leeds
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details
*/

/*!
  \file
  \ingroup projdata

  \brief  Non-inline implementations of
  stir::ProjDataInfoBlocksOnCylindricalNoArcCorr

  \author Kris Thielemans
  \author Palak Wadhwa
  \author Parisa Khateri

*/

#include "stir/ProjDataInfoBlocksOnCylindricalNoArcCorr.h"
#include "stir/Bin.h"
#include "stir/LORCoordinates.h"
#include "stir/round.h"
#include "stir/DetectionPosition.h"
#include <iostream>
#include <fstream>

#ifdef BOOST_NO_STRINGSTREAM
#include <strstream.h>
#else
#include <sstream>
#endif

#ifndef STIR_NO_NAMESPACES
using std::endl;
using std::ends;
#endif

START_NAMESPACE_STIR
ProjDataInfoBlocksOnCylindricalNoArcCorr::
ProjDataInfoBlocksOnCylindricalNoArcCorr()
{}

ProjDataInfoBlocksOnCylindricalNoArcCorr::
ProjDataInfoBlocksOnCylindricalNoArcCorr(const shared_ptr<Scanner> scanner_ptr,
                                 const  VectorWithOffset<int>& num_axial_pos_per_segment,
                                 const  VectorWithOffset<int>& min_ring_diff_v,
                                 const  VectorWithOffset<int>& max_ring_diff_v,
                                 const int num_views,const int num_tangential_poss)
: ProjDataInfoGenericNoArcCorr(scanner_ptr,
                          num_axial_pos_per_segment,
                          min_ring_diff_v, max_ring_diff_v,
                          num_views, num_tangential_poss)
{
  if (is_null_ptr(scanner_ptr))
    error("ProjDataInfoBlocksOnCylindricalNoArcCorr needs to be initialised with a non-empty Scanner");
  if (scanner_ptr->get_scanner_geometry() != "BlocksOnCylindrical")
    error("ProjDataInfoBlocksOnCylindricalNoArcCorr needs to be initialised with a Scanner with appropriate geometry");
}



ProjDataInfo*
ProjDataInfoBlocksOnCylindricalNoArcCorr::clone() const
{
  return static_cast<ProjDataInfo*>(new ProjDataInfoBlocksOnCylindricalNoArcCorr(*this));
}

bool
ProjDataInfoBlocksOnCylindricalNoArcCorr::
operator==(const self_type& that) const
{
  if (!base_type::blindly_equals(&that))
    return false;
  // TODOBLOCKS check crystal_map
  return
    true;
}

bool
ProjDataInfoBlocksOnCylindricalNoArcCorr::
blindly_equals(const root_type * const that_ptr) const
{
  assert(dynamic_cast<const self_type * const>(that_ptr) != 0);
  return
    this->operator==(static_cast<const self_type&>(*that_ptr));
}

std::string
ProjDataInfoBlocksOnCylindricalNoArcCorr::parameter_info()  const
{

 #ifdef BOOST_NO_STRINGSTREAM
  // dangerous for out-of-range, but 'old-style' ostrstream seems to need this
  char str[50000];
  ostrstream s(str, 50000);
 #else
  std::ostringstream s;
 #endif
  s << "ProjDataInfoBlocksOnCylindricalNoArcCorr := \n";
  s << base_type::parameter_info();
  s << "End :=\n";
  return s.str();
}

//!warning Use crystal map
Succeeded
ProjDataInfoBlocksOnCylindricalNoArcCorr::
find_scanner_coordinates_given_cartesian_coordinates(int& det1, int& det2, int& ring1, int& ring2,
					             const CartesianCoordinate3D<float>& c1,
						     	 const CartesianCoordinate3D<float>& c2) const
{

  DetectionPosition<> det_pos1;
  DetectionPosition<> det_pos2;
  if (get_scanner_ptr()->find_detection_position_given_cartesian_coordinate(det_pos1, c1+this->z_shift)==Succeeded::no ||
      get_scanner_ptr()->find_detection_position_given_cartesian_coordinate(det_pos2, c2+this->z_shift)==Succeeded::no)
  {
    return Succeeded::no;
  }

  det1 = det_pos1.tangential_coord();
  det2 = det_pos2.tangential_coord();
  ring1 = det_pos1.axial_coord();
  ring2 = det_pos2.axial_coord();

  assert(det1 >=0 && det1<get_scanner_ptr()->get_num_detectors_per_ring());
  assert(det2 >=0 && det2<get_scanner_ptr()->get_num_detectors_per_ring());

  return
    (ring1 >=0 && ring1<get_scanner_ptr()->get_num_rings() &&
     ring2 >=0 && ring2<get_scanner_ptr()->get_num_rings() &&
     det1!=det2)
     ? Succeeded::yes : Succeeded::no;
}

void
ProjDataInfoBlocksOnCylindricalNoArcCorr::
find_bin_given_cartesian_coordinates_of_detection(Bin& bin,
						  const CartesianCoordinate3D<float>& coord_1,
						  const CartesianCoordinate3D<float>& coord_2) const
{
  int det_num_a;
  int det_num_b;
  int ring_a;
  int ring_b;

  // given two CartesianCoordinates find the intersection
  if (find_scanner_coordinates_given_cartesian_coordinates(det_num_a,det_num_b,
							   ring_a, ring_b,
							   coord_1,
							   coord_2) ==
      Succeeded::no)
  {
    bin.set_bin_value(-1);
    return;
  }

  // check rings are in valid range
  // this should have been done by find_scanner_coordinates_given_cartesian_coordinates
  assert(!(ring_a<0 ||
	   ring_a>=get_scanner_ptr()->get_num_rings() ||
	   ring_b<0 ||
	   ring_b>=get_scanner_ptr()->get_num_rings()));

  if (get_bin_for_det_pair(bin,
			   det_num_a, ring_a,
			   det_num_b, ring_b) == Succeeded::no ||
      bin.tangential_pos_num() < get_min_tangential_pos_num() ||
      bin.tangential_pos_num() > get_max_tangential_pos_num())
    bin.set_bin_value(-1);
}

END_NAMESPACE_STIR
