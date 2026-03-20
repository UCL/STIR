/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2005, Hammersmith Imanet Ltd
    Copyright (C) 2013, University College London
    Copyright (C) 2013, Institute for Bioengineering of Catalonia
    Copyright (C) 2017, ETH Zurich, Institute of Particle Physics and Astrophysics
    Copyright (C) 2018, 2021, University of Leeds
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

/*!

  \file
  \ingroup projdata

  \brief Implementation of inline functions of class stir::ProjDataInfoGenericNoArcCorr

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author Palak Wadhwa
  \author Berta Marti Fuster
  \author PARAPET project
  \author Parisa Khateri
  \author Michael Roethlisberger
  \author Viet Ahn Dao
*/

// for sqrt
#include <cmath>
#include "stir/Bin.h"
#include "stir/Succeeded.h"
#include "stir/LORCoordinates.h"
#include "stir/is_null_ptr.h"
#include <algorithm>
#include "stir/CartesianCoordinate3D.h"
#include "stir/error.h"

START_NAMESPACE_STIR

//! find phi from correspoding lor
float
ProjDataInfoGenericNoArcCorr::get_phi(const Bin& bin) const
{
  LORInAxialAndNoArcCorrSinogramCoordinates<float> lor;
  get_LOR(lor, bin);
  return lor.phi();
}

/*! warning In generic geometry m is calculated directly from lor while in
        cylindrical geometry m is calculated using m_offset and axial_sampling
*/
float
ProjDataInfoGenericNoArcCorr::get_m(const Bin& bin) const
{
  LORInAxialAndNoArcCorrSinogramCoordinates<float> lor;
  get_LOR(lor, bin);
  return (static_cast<float>(lor.z1() + lor.z2())) / 2.F;
}

float
ProjDataInfoGenericNoArcCorr::get_t(const Bin& bin) const
{
  return get_m(bin) * get_costheta(bin);
}

/*
        theta is copolar angle of normal to projection plane with z axis, i.e. copolar angle of lor with z axis.
        tan (theta) = dz/sqrt(dx2+dy2)
        cylindrical geometry:
                delta_z = delta_ring * ring spacing
        generic geometry:
                delta_z is calculated from lor
*/
float
ProjDataInfoGenericNoArcCorr::get_tantheta(const Bin& bin) const
{
  LORInAxialAndNoArcCorrSinogramCoordinates<float> lor;
  get_LOR(lor, bin);

  return (lor.z2() - lor.z1()) / (2 * lor.radius());
}

float
ProjDataInfoGenericNoArcCorr::get_sampling_in_m(const Bin& bin) const
{
  // TODOBLOCK
  return get_scanner_ptr()
      ->get_ring_spacing(); // TODO currently restricted to span=1 get_num_axial_poss_per_ring_inc(segment_num);
  // return get_axial_sampling(bin.segment_num());
}

float
ProjDataInfoGenericNoArcCorr::get_sampling_in_t(const Bin& bin) const
{
  // TODOBLOCK
  return get_scanner_ptr()->get_ring_spacing()
         * get_costheta(bin); // TODO currently restricted to span=1 get_num_axial_poss_per_ring_inc(segment_num);
  // return get_axial_sampling(bin.segment_num())*get_costheta(bin);
}

float
ProjDataInfoGenericNoArcCorr::get_axial_sampling(int segment_num) const
{
  // TODOBLOCK should check sampling
  return get_ring_spacing(); // TODO currently restricted to span=1 get_num_axial_poss_per_ring_inc(segment_num);
}

bool
ProjDataInfoGenericNoArcCorr::axial_sampling_is_uniform() const
{
  // TODOBLOCK should check sampling
  return true;
}

float
ProjDataInfoGenericNoArcCorr::get_ring_spacing() const
{
  return get_scanner_ptr()->get_ring_spacing();
}

void
ProjDataInfoGenericNoArcCorr::initialise_uncompressed_view_tangpos_to_det1det2_if_not_done_yet() const
{
  // for efficiency reasons, use "Double-Checked-Locking(DCL) pattern" with OpenMP atomic operation
  // OpenMP v3.1 or later required
  // thanks to yohjp:
  // http://stackoverflow.com/questions/27975737/how-to-handle-cached-data-structures-with-multi-threading-e-g-openmp
#if defined(STIR_OPENMP) && _OPENMP >= 201012
  bool initialised;
#  pragma omp atomic read
  initialised = uncompressed_view_tangpos_to_det1det2_initialised;

  if (!initialised)
#endif
    {
#if defined(STIR_OPENMP)
#  pragma omp critical(PROJDATAINFOCYLINDRICALNOARCCORR_VIEWTANGPOS_TO_DETS)
#endif
      {
        if (!uncompressed_view_tangpos_to_det1det2_initialised)
          initialise_uncompressed_view_tangpos_to_det1det2();
      }
    }
}

void
ProjDataInfoGenericNoArcCorr::initialise_det1det2_to_uncompressed_view_tangpos_if_not_done_yet() const
{
  // as above
#if defined(STIR_OPENMP) && _OPENMP >= 201012
  bool initialised;
#  pragma omp atomic read
  initialised = det1det2_to_uncompressed_view_tangpos_initialised;

  if (!initialised)
#endif
    {
#if defined(STIR_OPENMP)
#  pragma omp critical(PROJDATAINFOCYLINDRICALNOARCCORR_DETS_TO_VIEWTANGPOS)
#endif
      {
        if (!det1det2_to_uncompressed_view_tangpos_initialised)
          initialise_det1det2_to_uncompressed_view_tangpos();
      }
    }
}

/*! warning In cylindrical s is found from bin: sin(beta) = sin(tang_pos*angular_increment)
        In block it is calculated directly from corresponding lor
*/
float
ProjDataInfoGenericNoArcCorr::get_s(const Bin& bin) const
{
  LORInAxialAndNoArcCorrSinogramCoordinates<float> lor;
  get_LOR(lor, bin);
  return lor.s();
}

void
ProjDataInfoGenericNoArcCorr::get_det_num_pair_for_view_tangential_pos_num(int& det1_num,
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
ProjDataInfoGenericNoArcCorr::get_view_tangential_pos_num_for_det_num_pair(int& view_num,
                                                                           int& tang_pos_num,
                                                                           const int det1_num,
                                                                           const int det2_num) const
{
  assert(det1_num != det2_num);
  this->initialise_det1det2_to_uncompressed_view_tangpos_if_not_done_yet();

  view_num = det1det2_to_uncompressed_view_tangpos[det1_num][det2_num].view_num / get_view_mashing_factor();
  tang_pos_num = det1det2_to_uncompressed_view_tangpos[det1_num][det2_num].tang_pos_num;
  return det1det2_to_uncompressed_view_tangpos[det1_num][det2_num].swap_detectors;
}

Succeeded
ProjDataInfoGenericNoArcCorr::get_bin_for_det_pair(
    Bin& bin, const int det_num1, const int ring_num1, const int det_num2, const int ring_num2) const
{
  if (get_view_tangential_pos_num_for_det_num_pair(bin.view_num(), bin.tangential_pos_num(), det_num1, det_num2))
    return get_segment_axial_pos_num_for_ring_pair(bin.segment_num(), bin.axial_pos_num(), ring_num1, ring_num2);
  else
    return get_segment_axial_pos_num_for_ring_pair(bin.segment_num(), bin.axial_pos_num(), ring_num2, ring_num1);
}

Succeeded
ProjDataInfoGenericNoArcCorr::get_bin_for_det_pos_pair(Bin& bin, const DetectionPositionPair<>& dp) const
{
  return get_bin_for_det_pair(
      bin, dp.pos1().tangential_coord(), dp.pos1().axial_coord(), dp.pos2().tangential_coord(), dp.pos2().axial_coord());
}

void
ProjDataInfoGenericNoArcCorr::get_det_pair_for_bin(
    int& det_num1, int& ring_num1, int& det_num2, int& ring_num2, const Bin& bin) const
{
  get_det_num_pair_for_view_tangential_pos_num(det_num1, det_num2, bin.view_num(), bin.tangential_pos_num());
  get_ring_pair_for_segment_axial_pos_num(ring_num1, ring_num2, bin.segment_num(), bin.axial_pos_num());
}

void
ProjDataInfoGenericNoArcCorr::get_det_pos_pair_for_bin(DetectionPositionPair<>& dp, const Bin& bin) const
{
  // lousy work around because types don't match TODO remove!
#if 1
  int t1 = dp.pos1().tangential_coord(), a1 = dp.pos1().axial_coord(), t2 = dp.pos2().tangential_coord(),
      a2 = dp.pos2().axial_coord();
  get_det_pair_for_bin(t1, a1, t2, a2, bin);
  dp.pos1().tangential_coord() = t1;
  dp.pos1().axial_coord() = a1;
  dp.pos2().tangential_coord() = t2;
  dp.pos2().axial_coord() = a2;

#else

  get_det_pair_for_bin(
      dp.pos1().tangential_coord(), dp.pos1().axial_coord(), dp.pos2().tangential_coord(), dp.pos2().axial_coord(), bin);
#endif
}

END_NAMESPACE_STIR
