/*
    Copyright (C) 2017 ETH Zurich, Institute of Particle Physics and Astrophysics
    Copyright (C) 2018, 2021, 2026, University College London
    Copyright (C) 2022, National Physical Laboratory
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details
*/

/*!

  \file
  \ingroup projdata

  \brief Implementation of non-inline functions of class stir::ProjDataInfoGenericNoArcCorr

  \author Kris Thielemans
  \author Parisa Khateri
  \author Michael Roethlisberger
  \author Daniel Deidda
*/

#include "stir/ProjDataInfoGenericNoArcCorr.h"
#include "stir/Bin.h"
#include "stir/LORCoordinates.h"
#include "stir/DetectionPosition.h"
#include "stir/error.h"
#include <algorithm>
#include <iostream>
#include <sstream>

using std::swap;

START_NAMESPACE_STIR
ProjDataInfoGenericNoArcCorr::ProjDataInfoGenericNoArcCorr()
{}

ProjDataInfoGenericNoArcCorr::ProjDataInfoGenericNoArcCorr(const shared_ptr<Scanner> scanner_sptr,
                                                           const VectorWithOffset<int>& num_axial_pos_per_segment,
                                                           const VectorWithOffset<int>& min_ring_diff_v,
                                                           const VectorWithOffset<int>& max_ring_diff_v,
                                                           const int num_views,
                                                           const int num_tangential_poss)
    : ProjDataInfoCylindricalNoArcCorr(
        scanner_sptr, num_axial_pos_per_segment, min_ring_diff_v, max_ring_diff_v, num_views, num_tangential_poss)
{
  if (!scanner_sptr)
    error("ProjDataInfoGenericNoArcCorr: first argument (scanner_ptr) is zero");
  if (num_tangential_poss > scanner_sptr->get_max_num_non_arccorrected_bins())
    error("ProjDataInfoGenericNoArcCorr: number of tangential positions exceeds the maximum number of non arc-corrected bins set "
          "for the scanner.");
  if (scanner_sptr->get_max_num_views() != num_views)
    error("ProjDataInfoGenericNoArcCorr: view mashing is not supported");

  // find shift between "new" centre-of-scanner and "old" centre-of-first-ring coordinate system
  this->z_shift.z() = this->get_scanner_ptr()->get_coordinate_for_det_pos(DetectionPosition<>(0, 0, 0)).z();
  this->z_shift.y() = 0;
  this->z_shift.x() = 0;
}

ProjDataInfo*
ProjDataInfoGenericNoArcCorr::clone() const
{
  return static_cast<ProjDataInfo*>(new ProjDataInfoGenericNoArcCorr(*this));
}

bool
ProjDataInfoGenericNoArcCorr::operator==(const self_type& that) const
{
  if (!base_type::blindly_equals(&that))
    return false;
  // TODO this is incomplete, probably
  return true;
}

bool
ProjDataInfoGenericNoArcCorr::blindly_equals(const root_type* const that_ptr) const
{
  assert(dynamic_cast<const self_type* const>(that_ptr) != 0);
  return this->operator==(static_cast<const self_type&>(*that_ptr));
}

void
ProjDataInfoGenericNoArcCorr::set_azimuthal_angle_offset(const float angle)
{
  if (angle != get_azimuthal_angle_offset())
    error("ProjDataInfoGenericNoArcCorr::set_azimuthal_angle_offset is not supported");
}

void
ProjDataInfoGenericNoArcCorr::set_azimuthal_angle_sampling(const float)
{
  error("ProjDataInfoGenericNoArcCorr::set_azimuthal_angle_sampling is not supported");
}

void
ProjDataInfoGenericNoArcCorr::set_ring_radii_for_all_views(const VectorWithOffset<float>&)
{
  error("ProjDataInfoGenericNoArcCorr::set_ring_radii_for_all_views is not supported");
}

void
ProjDataInfoGenericNoArcCorr::set_num_views(const int new_num_views)
{
  if (new_num_views != get_num_views())
    error("ProjDataInfoGenericNoArcCorr::set_num_views not supported");
}

void
ProjDataInfoGenericNoArcCorr::set_ring_spacing(float ring_spacing_v)
{
  if (ring_spacing_v != get_ring_spacing())
    error("ProjDataInfoGenericNoArcCorr::set_ring_spacing is not supported");
}

//! Find lor from cartesian coordinates of detector pair
void
ProjDataInfoGenericNoArcCorr::get_LOR(LORInAxialAndNoArcCorrSinogramCoordinates<float>& lor, const Bin& bin) const
{
  CartesianCoordinate3D<float> _p1;
  CartesianCoordinate3D<float> _p2;
  find_cartesian_coordinates_of_detection(_p1, _p2, bin);

  _p1.z() += z_shift.z();
  _p2.z() += z_shift.z();

  LORAs2Points<float> lor_as_2_points(_p1, _p2);
  const double R = sqrt(std::max(square(_p1.x()) + square(_p1.y()), square(_p2.x()) + square(_p2.y())));

  lor_as_2_points.change_representation(lor, R);
}

std::string
ProjDataInfoGenericNoArcCorr::parameter_info() const
{

  std::ostringstream s;

  s << "ProjDataInfoGenericNoArcCorr := \n";
  s << ProjDataInfo::parameter_info();
  // TODOBLOCK Cylindrical has the following which doesn't make sense for Generic, so repeat code
  // s << "Azimuthal angle increment (deg):   " << get_azimuthal_angle_sampling()*180/_PI << '\n';
  // s << "Azimuthal angle extent (deg):      " << fabs(get_azimuthal_angle_sampling())*get_num_views()*180/_PI << '\n';

  s << "ring differences per segment: \n";
  for (int segment_num = get_min_segment_num(); segment_num <= get_max_segment_num(); ++segment_num)
    {
      s << '(' << get_min_ring_difference(segment_num) << ',' << get_max_ring_difference(segment_num) << ')';
    }
  s << std::endl;
  s << "End :=\n";
  return s.str();
}

void
ProjDataInfoGenericNoArcCorr::find_cartesian_coordinates_given_scanner_coordinates(CartesianCoordinate3D<float>& coord_1,
                                                                                   CartesianCoordinate3D<float>& coord_2,
                                                                                   const int Ring_A,
                                                                                   const int Ring_B,
                                                                                   const int det1,
                                                                                   const int det2,
                                                                                   const int timing_pos_num) const
{
  assert(0 <= det1);
  assert(det1 < get_scanner_ptr()->get_num_detectors_per_ring());
  assert(0 <= det2);
  assert(det2 < get_scanner_ptr()->get_num_detectors_per_ring());

  DetectionPosition<> det_pos1;
  DetectionPosition<> det_pos2;
  det_pos1.tangential_coord() = det1;
  det_pos2.tangential_coord() = det2;
  det_pos1.axial_coord() = Ring_A;
  det_pos2.axial_coord() = Ring_B;

  coord_1 = get_scanner_ptr()->get_coordinate_for_det_pos(det_pos1);
  coord_2 = get_scanner_ptr()->get_coordinate_for_det_pos(det_pos2);
  coord_1.z() -= z_shift.z();
  coord_2.z() -= z_shift.z();

  if (timing_pos_num < 0)
    {
#ifndef ENABLE_TOF_GENERIC
      error("ProjDataInfoGenericNoArcCorr does not support TOF yet");
#endif
      // Currently timing_pos is unsigned, so we need to swap if the input is negative
      std::swap(coord_1, coord_2);
    }
}

/*!
  \warning \a lor currently has to be of type LORAs2Points<float> and points have to be in the
  detector map.
*/
Bin
ProjDataInfoGenericNoArcCorr::get_bin(const LOR<float>& lor, const double delta_time) const
{
#ifndef ENABLE_TOF_GENERIC
  if (delta_time != 0.)
    error("ProjDataInfoGenericNoArcCorr does not support TOF yet");
#endif

  Bin bin;

  if (!dynamic_cast<const LORAs2Points<float>*>(&lor))
    error("ProjDataInfoGenericNoArcCorr::get_bin(lor) currently only supports LORAs2Points");

  auto& lor_as_2points = static_cast<const LORAs2Points<float>&>(lor);
  auto _p1 = lor_as_2points.p1();
  auto _p2 = lor_as_2points.p2();

  DetectionPositionPair<> det_pos_pair;
  {
    const auto tof_bin = get_tof_bin(delta_time);
    // Currently timing_pos is unsigned, so we need to swap if the input is negative
    if (tof_bin < 0)
      {
        det_pos_pair.timing_pos() = -tof_bin;
        std::swap(_p1, _p2);
      }
    else
      {
        det_pos_pair.timing_pos() = tof_bin;
      }
  }
  auto& det_pos1 = det_pos_pair.pos1();
  auto& det_pos2 = det_pos_pair.pos2();

  if (get_scanner_ptr()->find_detection_position_given_cartesian_coordinate(det_pos1, _p1) == Succeeded::no
      || get_scanner_ptr()->find_detection_position_given_cartesian_coordinate(det_pos2, _p2) == Succeeded::no)
    {
      bin.set_bin_value(-1);
      return bin;
    }

  if (get_bin_for_det_pos_pair(bin, det_pos_pair) == Succeeded::yes && bin.tangential_pos_num() >= get_min_tangential_pos_num()
      && bin.tangential_pos_num() <= get_max_tangential_pos_num())
    {
      bin.set_bin_value(1);
      return bin;
    }
  else
    {
      bin.set_bin_value(-1);
      return bin;
    }
}

END_NAMESPACE_STIR
