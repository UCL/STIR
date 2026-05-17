
/*
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
#include "stir/DetectionPositionPair.h"
#include "stir/error.h"
#include <iostream>
#include <fstream>

#include <sstream>

using std::endl;
using std::ends;

START_NAMESPACE_STIR
ProjDataInfoBlocksOnCylindricalNoArcCorr::ProjDataInfoBlocksOnCylindricalNoArcCorr()
{}

ProjDataInfoBlocksOnCylindricalNoArcCorr::ProjDataInfoBlocksOnCylindricalNoArcCorr(
    const shared_ptr<Scanner> scanner_ptr,
    const VectorWithOffset<int>& num_axial_pos_per_segment,
    const VectorWithOffset<int>& min_ring_diff_v,
    const VectorWithOffset<int>& max_ring_diff_v,
    const int num_views,
    const int num_tangential_poss)
    : ProjDataInfoGenericNoArcCorr(
        scanner_ptr, num_axial_pos_per_segment, min_ring_diff_v, max_ring_diff_v, num_views, num_tangential_poss)
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
ProjDataInfoBlocksOnCylindricalNoArcCorr::operator==(const self_type& that) const
{
  if (!base_type::blindly_equals(&that))
    return false;
  // TODOBLOCKS check crystal_map
  return true;
}

bool
ProjDataInfoBlocksOnCylindricalNoArcCorr::blindly_equals(const root_type* const that_ptr) const
{
  assert(dynamic_cast<const self_type* const>(that_ptr) != 0);
  return this->operator==(static_cast<const self_type&>(*that_ptr));
}

std::string
ProjDataInfoBlocksOnCylindricalNoArcCorr::parameter_info() const
{

  std::ostringstream s;

  s << "ProjDataInfoBlocksOnCylindricalNoArcCorr := \n";
  s << base_type::parameter_info();
  s << "End :=\n";
  return s.str();
}

//! warning Use crystal map
Succeeded
ProjDataInfoBlocksOnCylindricalNoArcCorr::find_scanner_coordinates_given_cartesian_coordinates(
    int& det1, int& det2, int& ring1, int& ring2, const CartesianCoordinate3D<float>& c1, const CartesianCoordinate3D<float>& c2)
    const
{

  DetectionPosition<> det_pos1;
  DetectionPosition<> det_pos2;
  if (get_scanner_ptr()->find_detection_position_given_cartesian_coordinate(det_pos1, c1 + this->z_shift) == Succeeded::no
      || get_scanner_ptr()->find_detection_position_given_cartesian_coordinate(det_pos2, c2 + this->z_shift) == Succeeded::no)
    {
      return Succeeded::no;
    }

  det1 = det_pos1.tangential_coord();
  det2 = det_pos2.tangential_coord();
  ring1 = det_pos1.axial_coord();
  ring2 = det_pos2.axial_coord();

  assert(det1 >= 0 && det1 < get_scanner_ptr()->get_num_detectors_per_ring());
  assert(det2 >= 0 && det2 < get_scanner_ptr()->get_num_detectors_per_ring());

  return (ring1 >= 0 && ring1 < get_scanner_ptr()->get_num_rings() && ring2 >= 0 && ring2 < get_scanner_ptr()->get_num_rings()
          && det1 != det2)
             ? Succeeded::yes
             : Succeeded::no;
}

END_NAMESPACE_STIR
