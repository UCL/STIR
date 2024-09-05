/*
    Copyright (C) 2000- 2007-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2011, Kris Thielemans
    Copyright (C) 2017 ETH Zurich, Institute of Particle Physics and Astrophysics
    Copyright (C) 2018, University College London
    Copyright (C) 2018, University of Leeds
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details
*/

/*!

  \file
  \ingroup projdata

  \brief Implementation of non-inline functions of class stir::ProjDataInfoGenericNoArcCorr

  \author Kris Thielemans
  \author Palak Wadhwa
  \author Parisa Khateri
  \author Michael Roethlisberger
*/

#include "stir/ProjDataInfoGenericNoArcCorr.h"
#include "stir/Bin.h"
#include "stir/LORCoordinates.h"
#include "stir/round.h"
#include "stir/DetectionPosition.h"
#include "stir/is_null_ptr.h"
#include "stir/error.h"
#include <iostream>
#include <fstream>

#include <sstream>

#include <boost/static_assert.hpp>

using std::endl;
using std::ends;

START_NAMESPACE_STIR
ProjDataInfoGenericNoArcCorr::ProjDataInfoGenericNoArcCorr()
{}

ProjDataInfoGenericNoArcCorr::ProjDataInfoGenericNoArcCorr(const shared_ptr<Scanner> scanner_sptr,
                                                           const VectorWithOffset<int>& num_axial_pos_per_segment,
                                                           const VectorWithOffset<int>& min_ring_diff_v,
                                                           const VectorWithOffset<int>& max_ring_diff_v,
                                                           const int num_views,
                                                           const int num_tangential_poss)
    : ProjDataInfoGeneric(
        scanner_sptr, num_axial_pos_per_segment, min_ring_diff_v, max_ring_diff_v, num_views, num_tangential_poss)
{
  if (!scanner_sptr)
    error("ProjDataInfoGenericNoArcCorr: first argument (scanner_ptr) is zero");
  if (num_tangential_poss > scanner_sptr->get_max_num_non_arccorrected_bins())
    error("ProjDataInfoGenericNoArcCorr: number of tangential positions exceeds the maximum number of non arc-corrected bins set "
          "for the scanner.");
  if (scanner_sptr->get_max_num_views() != num_views)
    error("ProjDataInfoGenericNoArcCorr: view mashing is not supported");

  uncompressed_view_tangpos_to_det1det2_initialised = false;
  det1det2_to_uncompressed_view_tangpos_initialised = false;
#ifdef STIR_OPENMP_SAFE_BUT_SLOW
  this->initialise_uncompressed_view_tangpos_to_det1det2();
  this->initialise_det1det2_to_uncompressed_view_tangpos();
#endif

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
  return true;
}

bool
ProjDataInfoGenericNoArcCorr::blindly_equals(const root_type* const that_ptr) const
{
  assert(dynamic_cast<const self_type* const>(that_ptr) != 0);
  return this->operator==(static_cast<const self_type&>(*that_ptr));
}

std::string
ProjDataInfoGenericNoArcCorr::parameter_info() const
{

  std::ostringstream s;

  s << "ProjDataInfoGenericNoArcCorr := \n";
  s << ProjDataInfoGeneric::parameter_info();
  s << "End :=\n";
  return s.str();
}

/*
   TODO make compile time assert

   Warning:
   this code makes use of an implementation dependent feature:
   bit shifting negative ints to the right.
    -1 >> 1 should be -1
    -2 >> 1 should be -1
   This is ok on SUNs (gcc, but probably SUNs cc as well), Parsytec (gcc),
   Pentium (gcc, VC++) and probably every other system which uses
   the 2-complement convention.

   Update: compile time assert is implemented.
*/

/*!
  Go from sinograms to detectors.

  Because sinograms are not arc-corrected, tang_pos_num corresponds
  to an angle as well. Before interleaving we have that
  \verbatim
  det_angle_1 = LOR_angle + bin_angle
  det_angle_2 = LOR_angle + (Pi - bin_angle)
  \endverbatim
  (Hint: understand this first at LOR_angle=0, then realise that
  other LOR_angles follow just by rotation)

  Code gets slightly intricate because:
  - angles have to be defined modulo 2 Pi (so num_detectors)
  - interleaving
*/

//! build look-up table for get_view_tangential_pos_num_for_det_num_pair()
void
ProjDataInfoGenericNoArcCorr::initialise_uncompressed_view_tangpos_to_det1det2() const
{
  BOOST_STATIC_ASSERT(-1 >> 1 == -1);
  BOOST_STATIC_ASSERT(-2 >> 1 == -1);

  const int num_detectors = get_scanner_ptr()->get_num_detectors_per_ring();
  assert(num_detectors % 2 == 0);

  const int min_tang_pos_num = -(num_detectors / 2);
  const int max_tang_pos_num = -(num_detectors / 2) + num_detectors - 1;

  if (this->get_min_tangential_pos_num() < min_tang_pos_num || this->get_max_tangential_pos_num() > max_tang_pos_num)
    {
      error("The tangential_pos range (%d to %d) for this projection data is too large.\n"
            "Maximum supported range is from %d to %d",
            this->get_min_tangential_pos_num(),
            this->get_max_tangential_pos_num(),
            min_tang_pos_num,
            max_tang_pos_num);
    }

  uncompressed_view_tangpos_to_det1det2.grow(0, num_detectors / 2 - 1);
  for (int v_num = 0; v_num <= num_detectors / 2 - 1; ++v_num)
    {
      uncompressed_view_tangpos_to_det1det2[v_num].grow(min_tang_pos_num, max_tang_pos_num);

      for (int tp_num = min_tang_pos_num; tp_num <= max_tang_pos_num; ++tp_num)
        {
          /*
             adapted from CTI code
             Note for implementation: avoid using % with negative numbers
             so add num_detectors before doing modulo num_detectors)
            */
          uncompressed_view_tangpos_to_det1det2[v_num][tp_num].det1_num = (v_num + (tp_num >> 1) + num_detectors) % num_detectors;
          uncompressed_view_tangpos_to_det1det2[v_num][tp_num].det2_num
              = (v_num - ((tp_num + 1) >> 1) + num_detectors / 2) % num_detectors;
        }
    }
    // thanks to yohjp:
    // http://stackoverflow.com/questions/27975737/how-to-handle-cached-data-structures-with-multi-threading-e-g-openmp
#if defined(STIR_OPENMP) && _OPENMP >= 201012
#  pragma omp atomic write
#endif
  uncompressed_view_tangpos_to_det1det2_initialised = true;
}

void
ProjDataInfoGenericNoArcCorr::initialise_det1det2_to_uncompressed_view_tangpos() const
{
  BOOST_STATIC_ASSERT(-1 >> 1 == -1);
  BOOST_STATIC_ASSERT(-2 >> 1 == -1);

  const int num_detectors = get_scanner_ptr()->get_num_detectors_per_ring();

  if (num_detectors % 2 != 0)
    {
      error("Number of detectors per ring should be even but is %d", num_detectors);
    }
  if (this->get_min_view_num() != 0)
    {
      error("Minimum view number should currently be zero to be able to use get_view_tangential_pos_num_for_det_num_pair()");
    }

  // const int min_tang_pos_num = -(num_detectors/2);
  // const int max_tang_pos_num = -(num_detectors/2)+num_detectors;
  const int max_num_views = num_detectors / 2;

  det1det2_to_uncompressed_view_tangpos.grow(0, num_detectors - 1);
  for (int det1_num = 0; det1_num < num_detectors; ++det1_num)
    {
      det1det2_to_uncompressed_view_tangpos[det1_num].grow(0, num_detectors - 1);

      for (int det2_num = 0; det2_num < num_detectors; ++det2_num)
        {
          if (det1_num == det2_num)
            continue;
          /*
           This somewhat obscure formula was obtained by inverting the code for
           get_det_num_pair_for_view_tangential_pos_num()
           This can be simplified (especially all the branching later on), but
           as we execute this code only occasionally, it's probably not worth it.
          */
          int swap_detectors;
          /*
          Note for implementation: avoid using % with negative numbers
          so add num_detectors before doing modulo num_detectors
          */
          int tang_pos_num = (det1_num - det2_num + 3 * num_detectors / 2) % num_detectors;
          int view_num = (det1_num - (tang_pos_num >> 1) + num_detectors) % num_detectors;

          /* Now adjust ranges for view_num, tang_pos_num.
          The next lines go only wrong in the singular (and irrelevant) case
          det_num1 == det_num2 (when tang_pos_num == num_detectors - tang_pos_num)

            We use the combinations of the following 'symmetries' of
            (tang_pos_num, view_num) == (tang_pos_num+2*num_views, view_num + num_views)
            == (-tang_pos_num, view_num + num_views)
            Using the latter interchanges det_num1 and det_num2, and this leaves
            the LOR the same in the 2D case. However, in 3D this interchanges the rings
            as well. So, we keep track of this in swap_detectors, and return its final
            value.
          */
          if (view_num < max_num_views)
            {
              if (tang_pos_num >= max_num_views)
                {
                  tang_pos_num = num_detectors - tang_pos_num;
                  swap_detectors = 1;
                }
              else
                {
                  swap_detectors = 0;
                }
            }
          else
            {
              view_num -= max_num_views;
              if (tang_pos_num >= max_num_views)
                {
                  tang_pos_num -= num_detectors;
                  swap_detectors = 0;
                }
              else
                {
                  tang_pos_num *= -1;
                  swap_detectors = 1;
                }
            }

          det1det2_to_uncompressed_view_tangpos[det1_num][det2_num].view_num = view_num;
          det1det2_to_uncompressed_view_tangpos[det1_num][det2_num].tang_pos_num = tang_pos_num;
          det1det2_to_uncompressed_view_tangpos[det1_num][det2_num].swap_detectors = swap_detectors == 0;
        }
    }
    // thanks to yohjp:
    // http://stackoverflow.com/questions/27975737/how-to-handle-cached-data-structures-with-multi-threading-e-g-openmp
#if defined(STIR_OPENMP) && _OPENMP >= 201012
#  pragma omp atomic write
#endif
  det1det2_to_uncompressed_view_tangpos_initialised = true;
}

unsigned int
ProjDataInfoGenericNoArcCorr::get_num_det_pos_pairs_for_bin(const Bin& bin) const
{
  return get_num_ring_pairs_for_segment_axial_pos_num(bin.segment_num(), bin.axial_pos_num()) * get_view_mashing_factor();
}

void
ProjDataInfoGenericNoArcCorr::get_all_det_pos_pairs_for_bin(std::vector<DetectionPositionPair<>>& dps, const Bin& bin) const
{
  this->initialise_uncompressed_view_tangpos_to_det1det2_if_not_done_yet();

  dps.resize(get_num_det_pos_pairs_for_bin(bin));

  const ProjDataInfoGeneric::RingNumPairs& ring_pairs
      = get_all_ring_pairs_for_segment_axial_pos_num(bin.segment_num(), bin.axial_pos_num());
  // not sure how to handle mashing with non-zero view offset...
  assert(get_min_view_num() == 0);

  unsigned int current_dp_num = 0;
  for (int uncompressed_view_num = bin.view_num() * get_view_mashing_factor();
       uncompressed_view_num < (bin.view_num() + 1) * get_view_mashing_factor();
       ++uncompressed_view_num)
    {
      const int det1_num = uncompressed_view_tangpos_to_det1det2[uncompressed_view_num][bin.tangential_pos_num()].det1_num;
      const int det2_num = uncompressed_view_tangpos_to_det1det2[uncompressed_view_num][bin.tangential_pos_num()].det2_num;
      for (ProjDataInfoGeneric::RingNumPairs::const_iterator rings_iter = ring_pairs.begin(); rings_iter != ring_pairs.end();
           ++rings_iter)
        {
          assert(current_dp_num < get_num_det_pos_pairs_for_bin(bin));
          dps[current_dp_num].pos1().tangential_coord() = det1_num;
          dps[current_dp_num].pos1().axial_coord() = rings_iter->first;
          dps[current_dp_num].pos2().tangential_coord() = det2_num;
          dps[current_dp_num].pos2().axial_coord() = rings_iter->second;
          ++current_dp_num;
        }
    }
  assert(current_dp_num == get_num_det_pos_pairs_for_bin(bin));
}

void
ProjDataInfoGenericNoArcCorr::find_cartesian_coordinates_of_detection(CartesianCoordinate3D<float>& coord_1,
                                                                      CartesianCoordinate3D<float>& coord_2,
                                                                      const Bin& bin) const
{
  // find detectors
  int det_num_a;
  int det_num_b;
  int ring_a;
  int ring_b;
  get_det_pair_for_bin(det_num_a, ring_a, det_num_b, ring_b, bin);

  // find corresponding cartesian coordinates
  find_cartesian_coordinates_given_scanner_coordinates(coord_1, coord_2, ring_a, ring_b, det_num_a, det_num_b);
  return;
}

void
ProjDataInfoGenericNoArcCorr::find_cartesian_coordinates_given_scanner_coordinates(CartesianCoordinate3D<float>& coord_1,
                                                                                   CartesianCoordinate3D<float>& coord_2,
                                                                                   const int Ring_A,
                                                                                   const int Ring_B,
                                                                                   const int det1,
                                                                                   const int det2) const
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
}

Bin
ProjDataInfoGenericNoArcCorr::get_bin(const LOR<float>& lor, const double delta_time) const
{
  if (delta_time != 0.)
    error("ProjDataInfoGenericNoArcCorr does not support TOF yet");

  Bin bin;

  const LORAs2Points<float>& lor_as_2points = dynamic_cast<const LORAs2Points<float>&>(lor);

  CartesianCoordinate3D<float> _p1 = lor_as_2points.p1();
  CartesianCoordinate3D<float> _p2 = lor_as_2points.p2();

  DetectionPosition<> det_pos1;
  DetectionPosition<> det_pos2;

  if (get_scanner_ptr()->find_detection_position_given_cartesian_coordinate(det_pos1, _p1) == Succeeded::no
      || get_scanner_ptr()->find_detection_position_given_cartesian_coordinate(det_pos2, _p2) == Succeeded::no)
    {
      bin.set_bin_value(-1);
      return bin;
    }

  DetectionPositionPair<> det_pos_pair;
  det_pos_pair.pos1() = det_pos1;
  det_pos_pair.pos2() = det_pos2;

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
