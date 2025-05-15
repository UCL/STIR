/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2009-10-18 Hammersmith Imanet Ltd
    Copyright (C) 2011, Kris Thielemans
    Copyright (C) 2013, 2017, 2018, 2021, 2022 University College London
    Copyright (C) 2016, University of Hull

    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup projdata

  \brief Non-inline implementations of stir::ProjDataInfoCylindrical

  \author Nikos Efthimiou
  \author Kris Thielemans
  \author Sanida Mustafovic
  \author PARAPET project
*/

#include "stir/ProjDataInfoCylindrical.h"
#include "stir/LORCoordinates.h"
#include "stir/Array.h"
#include <algorithm>
#include <sstream>

#include "stir/round.h"
#include "stir/numerics/norm.h"
#include "stir/warning.h"
#include "stir/info.h"
#include "stir/format.h"
#include <math.h>
#include "stir/warning.h"
#include "stir/error.h"

using std::min_element;
using std::max_element;
using std::min;
using std::max;
using std::endl;
using std::string;
using std::pair;
using std::vector;

START_NAMESPACE_STIR

ProjDataInfoCylindrical::ProjDataInfoCylindrical()
{}

ProjDataInfoCylindrical::ProjDataInfoCylindrical(const shared_ptr<Scanner>& scanner_ptr,
                                                 const VectorWithOffset<int>& num_axial_pos_per_segment,
                                                 const VectorWithOffset<int>& min_ring_diff_v,
                                                 const VectorWithOffset<int>& max_ring_diff_v,
                                                 const int num_views,
                                                 const int num_tangential_poss)
    : ProjDataInfo(scanner_ptr, num_axial_pos_per_segment, num_views, num_tangential_poss),
      min_ring_diff(min_ring_diff_v),
      max_ring_diff(max_ring_diff_v)
{
  azimuthal_angle_sampling = static_cast<float>(_PI / num_views);
  azimuthal_angle_offset = scanner_ptr->get_intrinsic_azimuthal_tilt();
  // adjust offset for view-mashing
  {
    const int num_detectors_per_ring = scanner_ptr->get_num_detectors_per_ring();
    if ((num_detectors_per_ring > 2) && (num_views * 2 != num_detectors_per_ring))
      {
        if ((num_detectors_per_ring % (num_views * 2)) != 0)
          {
            warning(format("Expected the number of views ({}) to be related to the number of detectors per ring ({}),"
                           " but this is not the case. Continuing anyway (but without adjusting the azimuthal angle offset).",
                           num_views,
                           num_detectors_per_ring));
          }
        else
          {
            const int view_mashing = get_view_mashing_factor();
            const float offset_inc = static_cast<float>(_PI / (num_detectors_per_ring / 2) * (view_mashing - 1) / 2.F);
            info(format("Detected view-mashing factor {} from the number of views ({}) and the number of detectors per "
                        "ring ({}).\n"
                        "Adjusting the azimuthal angle offset accordingly (an extra offset of {} degrees)",
                        view_mashing,
                        num_views,
                        num_detectors_per_ring,
                        (offset_inc * 180 / _PI)));

            azimuthal_angle_offset += offset_inc;
          }
      }
  }

  ring_radius.resize(0, 0);
  ring_radius[0] = get_scanner_ptr()->get_effective_ring_radius();
  ring_spacing = get_scanner_ptr()->get_ring_spacing();

  // TODO this info should probably be provided via the constructor, or at
  // least by Scanner.
  sampling_corresponds_to_physical_rings = scanner_ptr->get_type() != Scanner::HiDAC;

  assert(min_ring_diff.get_length() == max_ring_diff.get_length());
  assert(min_ring_diff.get_length() == num_axial_pos_per_segment.get_length());

  // check min,max ring diff
  {
    for (int segment_num = get_min_segment_num(); segment_num <= get_max_segment_num(); ++segment_num)
      if (min_ring_diff[segment_num] > max_ring_diff[segment_num])
        {
          warning(format("ProjDataInfoCylindrical: min_ring_difference {} is larger than max_ring_difference {} for segment {}. "
                         "Swapping them around",
                         min_ring_diff[segment_num],
                         max_ring_diff[segment_num],
                         segment_num));
          std::swap(min_ring_diff[segment_num], max_ring_diff[segment_num]);
        }
  }

  initialise_ring_diff_arrays();
}

void
ProjDataInfoCylindrical::initialise_ring_diff_arrays() const
{

  // check min,max ring diff
  {
    // check is necessary here again because of set_min_ring_difference()
    // we do not swap here because that would require the min/max_ring_diff arrays to be mutable as well
    for (int segment_num = get_min_segment_num(); segment_num <= get_max_segment_num(); ++segment_num)
      if (min_ring_diff[segment_num] > max_ring_diff[segment_num])
        {
          error("ProjDataInfoCylindrical: min_ring_difference %d is larger than "
                "max_ring_difference %d for segment %d.",
                min_ring_diff[segment_num],
                max_ring_diff[segment_num],
                segment_num);
        }
  }
  // initialise m_offset
  {
    m_offset = VectorWithOffset<float>(get_min_segment_num(), get_max_segment_num());

    /* m_offsets are found by requiring
    get_m(..., min_axial_pos_num,...) == - get_m(..., max_axial_pos_num,...)
    */
    for (int segment_num = get_min_segment_num(); segment_num <= get_max_segment_num(); ++segment_num)
      {
        m_offset[segment_num]
            = ((get_max_axial_pos_num(segment_num) + get_min_axial_pos_num(segment_num)) * get_axial_sampling(segment_num)) / 2;
      }
  }
  // initialise ax_pos_num_offset
  if (sampling_corresponds_to_physical_rings)
    {
      const int num_rings = get_scanner_ptr()->get_num_rings();
      ax_pos_num_offset = VectorWithOffset<int>(get_min_segment_num(), get_max_segment_num());

      /* ax_pos_num will be determined by looking at ring1+ring2.
         This also works for axially compressed data (i.e. span) as
         ring1+ring2 is constant for all ring-pairs combined into 1
         segment,ax_pos.

         Ignoring the difficulties of axial compression for a second, it is clear that
         for a given bin, there will be 2 rings as follows:
           ring2 = get_m(bin)/ring_spacing  + ring_diff/2 + (num_rings-1)/2
           ring1 = get_m(bin)/ring_spacing  - ring_diff/2 + (num_rings-1)/2
         This follows from the fact that get_m() returns the z position
         in millimeter of the middle of the LOR w.r.t. the middle of the scanner.
         The (num_rings-1)/2 shifts the origin such that the first ring has
         ring_num==0.

         From the above, it follows that
           ring1+ring2=2*get_m(bin)/ring_spacing + (num_rings-1)
         Finally, we use the formula for get_m to obtain
           ring1+ring2=2*ax_pos_num/get_num_axial_poss_per_ring_inc(segment_num)
                       -2*m_offset[segment_num]/ring_spacing + (num_rings-1)
         Solving this for ax_pos_num:
           ax_pos_num = (ring1+ring2-(num_rings-1)
                         + 2*m_offset[segment_num]/ring_spacing
                        ) * get_num_axial_poss_per_ring_inc(segment_num)/2

         We could plug m_offset in to obtain
           ax_pos_num = (ring1+ring2-(num_rings-1)
                        ) * get_num_axial_poss_per_ring_inc(segment_num)/2.
                        +
                        (get_max_axial_pos_num(segment_num)
                          + get_min_axial_pos_num(segment_num) )/2.
         this formula is easy to understand, but we don't use it as
         at some point somebody might change m_offset
         and forget to change this code...
         (also, the form above would need float division and then rounding)
         */
      for (int segment_num = get_min_segment_num(); segment_num <= get_max_segment_num(); ++segment_num)
        {
          ax_pos_num_offset[segment_num] = round((num_rings - 1) - 2 * m_offset[segment_num] / ring_spacing);
          if (get_scanner_sptr()->get_scanner_geometry() == "Cylindrical")
            {
              // check that it was integer
              if (fabs(ax_pos_num_offset[segment_num] - ((num_rings - 1) - 2 * m_offset[segment_num] / ring_spacing)) > 1E-4)
                {
                  error(format("ProjDataInfoCylindrical: in segment {}, the axial positions\n"
                               "do not correspond to the usual locations between physical rings.\n"
                               "This is suspicious and can make things go wrong in STIR, so I abort.\n"
                               "Check the number of axial positions in this segment.",
                               segment_num));
                }
            }

          if (get_num_axial_poss_per_ring_inc(segment_num) == 1)
            {
              // check that we'll get an integer ax_pos_num, i.e.
              // (ring1+ring2  - ax_pos_num_offset) has to be even, for any
              // ring1,ring2 in the segment, i.e ring1-ring2 = ring_diff, so
              // ring1+ring2 = 2*ring2 + ring_diff
              assert(get_min_ring_difference(segment_num) == get_max_ring_difference(segment_num));
              if ((get_max_ring_difference(segment_num) - ax_pos_num_offset[segment_num]) % 2 != 0)
                warning(format("ProjDataInfoCylindrical: the number of axial positions {} in "
                               "segment {} (ring_diff {}) is such that current conventions will place "
                               "the LORs shifted with respect to the physical rings {}.",
                               get_num_axial_poss(segment_num),
                               segment_num,
                               get_min_ring_difference(segment_num), // equal to max here, as per the if()
                               get_scanner_ptr()->get_num_rings()));
            }
        }
    }
  // initialise ring_diff_to_segment_num
  if (sampling_corresponds_to_physical_rings)
    {
      const int min_ring_difference = *min_element(min_ring_diff.begin(), min_ring_diff.end());
      const int max_ring_difference = *max_element(max_ring_diff.begin(), max_ring_diff.end());

      // set ring_diff_to_segment_num to appropriate size
      // in principle, the max ring difference would be scanner.num_rings-1, but
      // in case someone is up to strange things, we take the max of this value
      // with the max_ring_difference as given in the file
      ring_diff_to_segment_num = VectorWithOffset<int>(min(min_ring_difference, -(get_scanner_ptr()->get_num_rings() - 1)),
                                                       max(max_ring_difference, get_scanner_ptr()->get_num_rings() - 1));
      // first set all to impossible value
      // warning: get_segment_num_for_ring_difference relies on the fact that this value
      // is larger than get_max_segment_num()
      ring_diff_to_segment_num.fill(get_max_segment_num() + 1);

      for (int ring_diff = min_ring_difference; ring_diff <= max_ring_difference; ++ring_diff)
        {
          int segment_num;
          for (segment_num = get_min_segment_num(); segment_num <= get_max_segment_num(); ++segment_num)
            {
              if (ring_diff >= min_ring_diff[segment_num] && ring_diff <= max_ring_diff[segment_num])
                {
#if 0
          std::cerr << "ring diff " << ring_diff << " stored in s:" << segment_num << std::endl;
#endif
                  ring_diff_to_segment_num[ring_diff] = segment_num;
                  break;
                }
            }
          if (segment_num > get_max_segment_num())
            {
              warning(format("ProjDataInfoCylindrical: ring difference {} does not belong to a segment", ring_diff));
            }
        }
    }
  // initialise segment_axial_pos_to_ring1_plus_ring2
  if (sampling_corresponds_to_physical_rings)
    {
      segment_axial_pos_to_ring1_plus_ring2
          = VectorWithOffset<VectorWithOffset<int>>(get_min_segment_num(), get_max_segment_num());
      for (int s_num = get_min_segment_num(); s_num <= get_max_segment_num(); ++s_num)
        {
          const int min_ax_pos_num = get_min_axial_pos_num(s_num);
          const int max_ax_pos_num = get_max_axial_pos_num(s_num);
          segment_axial_pos_to_ring1_plus_ring2[s_num].grow(min_ax_pos_num, max_ax_pos_num);
          for (int ax_pos_num = min_ax_pos_num; ax_pos_num <= max_ax_pos_num; ++ax_pos_num)
            {
              // see documentation above for formulas
              const float ring1_plus_ring2_float = 2 * ax_pos_num / get_num_axial_poss_per_ring_inc(s_num)
                                                   - 2 * m_offset[s_num] / ring_spacing
                                                   + (get_scanner_ptr()->get_num_rings() - 1);
              const int ring1_plus_ring2 = round(ring1_plus_ring2_float);
              // check that it was integer
              if (get_scanner_sptr()->get_scanner_geometry() == "Cylindrical")
                {
                  assert(fabs(ring1_plus_ring2 - ring1_plus_ring2_float) < 1E-4);
                }
              segment_axial_pos_to_ring1_plus_ring2[s_num][ax_pos_num] = ring1_plus_ring2;
            }
        }
    }

  // now also initialise the segment_axial_pos_to_ring_pair table
  if (sampling_corresponds_to_physical_rings)
    {
      allocate_segment_axial_pos_to_ring_pair();

      for (int s_num = get_min_segment_num(); s_num <= get_max_segment_num(); ++s_num)
        {
          const int min_ax_pos_num = get_min_axial_pos_num(s_num);
          const int max_ax_pos_num = get_max_axial_pos_num(s_num);
          for (int ax_pos_num = min_ax_pos_num; ax_pos_num <= max_ax_pos_num; ++ax_pos_num)
            {
              compute_segment_axial_pos_to_ring_pair(s_num, ax_pos_num);
            }
        }
    }

  // Now make sure we never do this again...
  // Note that this variable needs to be set at *the end* of this function to make
  // everything thread-safe. Otherwise, with the double-locked loop paradigm that
  // we use, another thread can check the value of the variable before this function
  // is done.
  ring_diff_arrays_computed = true;
}

/*! Default implementation checks common variables. Needs to be overloaded.
 */
bool
ProjDataInfoCylindrical::blindly_equals(const root_type* const that) const
{
  if (!base_type::blindly_equals(that))
    return false;

  const self_type& proj_data_info = static_cast<const self_type&>(*that);
  const Array<1, float> tmp(this->ring_radius - proj_data_info.ring_radius);
  return fabs(this->azimuthal_angle_sampling - proj_data_info.azimuthal_angle_sampling) < 0.05F && norm(tmp) < 0.05F
         && this->sampling_corresponds_to_physical_rings == proj_data_info.sampling_corresponds_to_physical_rings
         && fabs(this->ring_spacing - proj_data_info.ring_spacing) < 0.05F && this->min_ring_diff == proj_data_info.min_ring_diff
         && this->max_ring_diff == proj_data_info.max_ring_diff;
}

void
ProjDataInfoCylindrical::get_ring_pair_for_segment_axial_pos_num(int& ring1,
                                                                 int& ring2,
                                                                 const int segment_num,
                                                                 const int axial_pos_num) const
{
  if (!sampling_corresponds_to_physical_rings)
    error("ProjDataInfoCylindrical::get_ring_pair_for_segment_axial_pos_num does not work for this type of sampled data");
  // can do only span=1 at the moment
  if (get_min_ring_difference(segment_num) != get_max_ring_difference(segment_num))
    error("ProjDataInfoCylindrical::get_ring_pair_for_segment_axial_pos_num does not work for data with axial compression");

  this->initialise_ring_diff_arrays_if_not_done_yet();

  const int ring_diff = get_max_ring_difference(segment_num);
  const int ring1_plus_ring2 = segment_axial_pos_to_ring1_plus_ring2[segment_num][axial_pos_num];

  // KT 01/08/2002 swapped rings
  ring1 = (ring1_plus_ring2 - ring_diff) / 2;
  ring2 = (ring1_plus_ring2 + ring_diff) / 2;
  assert((ring1_plus_ring2 + ring_diff) % 2 == 0);
  assert((ring1_plus_ring2 - ring_diff) % 2 == 0);
}

void
ProjDataInfoCylindrical::set_azimuthal_angle_offset(const float angle_v)
{
  azimuthal_angle_offset = angle_v;
}

void
ProjDataInfoCylindrical::set_azimuthal_angle_sampling(const float angle_v)
{
  azimuthal_angle_sampling = angle_v;
}

// void
// ProjDataInfoCylindrical::
// set_axial_sampling(const float samp_v, int segment_num)
//{axial_sampling = samp_v;}

void
ProjDataInfoCylindrical::set_num_views(const int new_num_views)
{
  const float old_azimuthal_angle_range = this->get_azimuthal_angle_sampling() * this->get_num_views();
  base_type::set_num_views(new_num_views);
  this->azimuthal_angle_sampling = old_azimuthal_angle_range / this->get_num_views();
}

void
ProjDataInfoCylindrical::set_min_ring_difference(int min_ring_diff_v, int segment_num)
{
  ring_diff_arrays_computed = false;
  min_ring_diff[segment_num] = min_ring_diff_v;
}

void
ProjDataInfoCylindrical::set_max_ring_difference(int max_ring_diff_v, int segment_num)
{
  ring_diff_arrays_computed = false;
  max_ring_diff[segment_num] = max_ring_diff_v;
}

void
ProjDataInfoCylindrical::set_ring_spacing(float ring_spacing_v)
{
  ring_diff_arrays_computed = false;
  ring_spacing = ring_spacing_v;
}

void
ProjDataInfoCylindrical::allocate_segment_axial_pos_to_ring_pair() const
{
  segment_axial_pos_to_ring_pair
      = VectorWithOffset<VectorWithOffset<shared_ptr<RingNumPairs>>>(get_min_segment_num(), get_max_segment_num());

  for (int segment_num = get_min_segment_num(); segment_num <= get_max_segment_num(); ++segment_num)
    {
      segment_axial_pos_to_ring_pair[segment_num].grow(get_min_axial_pos_num(segment_num), get_max_axial_pos_num(segment_num));
    }
}

void
ProjDataInfoCylindrical::compute_segment_axial_pos_to_ring_pair(const int segment_num, const int axial_pos_num) const
{
  shared_ptr<RingNumPairs> new_el(new RingNumPairs);
  segment_axial_pos_to_ring_pair[segment_num][axial_pos_num] = new_el;

  RingNumPairs& table = *segment_axial_pos_to_ring_pair[segment_num][axial_pos_num];
  table.reserve(get_max_ring_difference(segment_num) - get_min_ring_difference(segment_num) + 1);

  /* We compute the lookup-table in a fancy way.
     We could just as well have a simple loop over all ring pairs and check
     if it belongs to this segment/axial_pos.
     The current way is a lot faster though.
  */
  const int min_ring_diff = get_min_ring_difference(segment_num);
  const int max_ring_diff = get_max_ring_difference(segment_num);
  const int num_rings = get_scanner_ptr()->get_num_rings();

  /* ring1_plus_ring2 is the same for any ring pair that contributes to
     this particular segment_num, axial_pos_num.
  */
  const int ring1_plus_ring2 = segment_axial_pos_to_ring1_plus_ring2[segment_num][axial_pos_num];

  /*
    The ring_difference increments with 2 as the other ring differences do
    not give a ring pair with this axial_position. This is because
    ring1_plus_ring2%2 == ring_diff%2
    (which easily follows by plugging in ring1+ring2 and ring1-ring2).
    The starting ring_diff is determined such that the above condition
    is satisfied. You can check it by noting that the
      start_ring_diff%2
        == (min_ring_diff + (min_ring_diff+ring1_plus_ring2)%2)%2
        == (2*min_ring_diff+ring1_plus_ring2)%2
        == ring1_plus_ring2%2
  */
  for (int ring_diff = min_ring_diff + (min_ring_diff + ring1_plus_ring2) % 2; ring_diff <= max_ring_diff; ring_diff += 2)
    {
      const int ring1 = (ring1_plus_ring2 - ring_diff) / 2;
      const int ring2 = (ring1_plus_ring2 + ring_diff) / 2;
      if (ring1 < 0 || ring2 < 0 || ring1 >= num_rings || ring2 >= num_rings)
        continue;
      assert((ring1_plus_ring2 + ring_diff) % 2 == 0);
      assert((ring1_plus_ring2 - ring_diff) % 2 == 0);
      table.push_back(pair<int, int>(ring1, ring2));
    }
}

void
ProjDataInfoCylindrical::set_tof_mash_factor(const int new_num)
{
  base_type::set_tof_mash_factor(new_num);
  //! \todo N.E. Would be nice to have all the points of the scanner in cache.
  // initialise_uncompressed_lor_as_point1point2();
}

void
ProjDataInfoCylindrical::set_num_axial_poss_per_segment(const VectorWithOffset<int>& num_axial_poss_per_segment)
{
  ProjDataInfo::set_num_axial_poss_per_segment(num_axial_poss_per_segment);
  ring_diff_arrays_computed = false;
}

void
ProjDataInfoCylindrical::set_min_axial_pos_num(const int min_ax_pos_num, const int segment_num)
{
  ProjDataInfo::set_min_axial_pos_num(min_ax_pos_num, segment_num);
  ring_diff_arrays_computed = false;
}

void
ProjDataInfoCylindrical::set_max_axial_pos_num(const int max_ax_pos_num, const int segment_num)
{
  ProjDataInfo::set_max_axial_pos_num(max_ax_pos_num, segment_num);
  ring_diff_arrays_computed = false;
}

void
ProjDataInfoCylindrical::reduce_segment_range(const int min_segment_num, const int max_segment_num)
{
  ProjDataInfo::reduce_segment_range(min_segment_num, max_segment_num);
  // reduce ring_diff arrays to new valid size
  VectorWithOffset<int> new_min_ring_diff(min_segment_num, max_segment_num);
  VectorWithOffset<int> new_max_ring_diff(min_segment_num, max_segment_num);

  for (int segment_num = min_segment_num; segment_num <= max_segment_num; ++segment_num)
    {
      new_min_ring_diff[segment_num] = this->min_ring_diff[segment_num];
      new_max_ring_diff[segment_num] = this->max_ring_diff[segment_num];
    }

  this->min_ring_diff = new_min_ring_diff;
  this->max_ring_diff = new_max_ring_diff;

  // make sure other arrays will be updated if/when necessary
  this->ring_diff_arrays_computed = false;
}

void
ProjDataInfoCylindrical::get_LOR(LORInAxialAndNoArcCorrSinogramCoordinates<float>& lor, const Bin& bin) const
{
  const float s_in_mm = get_s(bin);
  const float m_in_mm = get_m(bin);
  const float tantheta = get_tantheta(bin);
  const float phi = get_phi(bin);
  /* parametrisation of LOR is
     X= s*cphi + a*sphi,
     Y= s*sphi - a*cphi,
     Z= m - a*tantheta
     find now min_a, max_a such that end-points intersect the ring
  */
  assert(fabs(s_in_mm) < get_ring_radius());
  // a has to be such that X^2+Y^2 == R^2
  const float max_a = sqrt(square(get_ring_radius()) - square(s_in_mm));
  const float min_a = -max_a;

  /*    start_point.x() = (s_in_mm*cphi + max_a*sphi);
        start_point.y() = (s_in_mm*sphi - max_a*cphi);
        start_point.z() = (m_in_mm - max_a*tantheta);
        stop_point.x() = (s_in_mm*cphi + min_a*sphi);
        stop_point.y() = (s_in_mm*sphi - min_a*cphi);
        stop_point.z() = (m_in_mm - min_a*tantheta);
  */
  const float z1 = (m_in_mm - max_a * tantheta);
  const float z2 = (m_in_mm - min_a * tantheta);

  lor = LORInAxialAndNoArcCorrSinogramCoordinates<float>(z1,
                                                         z2,
                                                         phi,
                                                         asin(s_in_mm / get_ring_radius()),
                                                         get_ring_radius(),
                                                         false); // needs to set "swapped" to false given above code
}

#if 0
  // KT disabled these as untested (and unused)

void
ProjDataInfoCylindrical::
get_LOR_as_two_points(CartesianCoordinate3D<float>& coord_1,
                      CartesianCoordinate3D<float>& coord_2,
                      const Bin& bin) const
{
    const float s_in_mm = get_s(bin);
    const float m_in_mm = get_m(bin);
    const float tantheta = get_tantheta(bin);
    const float phi = get_phi(bin);
    /* parametrisation of LOR is
     X= s*cphi + a*sphi,
     Y= s*sphi - a*cphi,
     Z= m - a*tantheta
     find now min_a, max_a such that end-points intersect the ring
  */
    assert(fabs(s_in_mm) < get_ring_radius());
    // a has to be such that X^2+Y^2 == R^2
    const float  max_a = sqrt(square(get_ring_radius()) - square(s_in_mm));
    const float  min_a = -max_a;

    coord_1.x() = s_in_mm*cos(phi) + min_a*sin(phi);
    coord_1.y() = s_in_mm*sin(phi) - max_a*cos(phi);
    coord_1.z() = m_in_mm - max_a*tantheta;

    coord_2.x() = s_in_mm*cos(phi) + max_a*sin(phi);
    coord_2.y() = s_in_mm*sin(phi) - min_a*cos(phi);
    coord_2.z() = m_in_mm - min_a*tantheta;

    if (bin.timing_pos_num()<0)
    	std::swap(coord_1, coord_2);
}

void
ProjDataInfoCylindrical::
get_LOR_as_two_points_alt(CartesianCoordinate3D<float>& coord_1,
                          CartesianCoordinate3D<float>& coord_2,
                          const int det1,
                          const int det2,
                          const int ring1,
                          const int ring2,
                          const int timing_pos) const
{
    const int num_detectors_per_ring =
            get_scanner_ptr()->get_num_detectors_per_ring();

    float h_scanner_height = ( (get_scanner_ptr()->get_ring_spacing() -1) * get_scanner_ptr()->get_num_rings())/2.F;

    // although code maybe doesn't really need the following,
    // asserts in the LOR code will break if these conditions are not satisfied.
    assert(0<=det1);
    assert(det1<num_detectors_per_ring);
    assert(0<=det2);
    assert(det2<num_detectors_per_ring);

    LORInCylinderCoordinates<float> cyl_coords(get_scanner_ptr()->get_inner_ring_radius());

    cyl_coords.p1().psi() = static_cast<float>((2.*_PI/num_detectors_per_ring)*(det1));
    cyl_coords.p2().psi() = static_cast<float>((2.*_PI/num_detectors_per_ring)*(det2));

    cyl_coords.p1().z() = ring1*get_scanner_ptr()->get_ring_spacing() - h_scanner_height;
    cyl_coords.p2().z() = ring2*get_scanner_ptr()->get_ring_spacing() - h_scanner_height;

    LORAs2Points<float> lor(cyl_coords);
    coord_1 = lor.p1();
    coord_2 = lor.p2();

	if (timing_pos<0)
		std::swap(coord_1, coord_2);
}
#endif

string
ProjDataInfoCylindrical::parameter_info() const
{

  std::ostringstream s;

  s << ProjDataInfo::parameter_info();
  s << "Azimuthal angle increment (deg):   " << get_azimuthal_angle_sampling() * 180 / _PI << '\n';
  s << "Azimuthal angle extent (deg):      " << fabs(get_azimuthal_angle_sampling()) * get_num_views() * 180 / _PI << '\n';

  s << "ring differences per segment: \n";
  for (int segment_num = get_min_segment_num(); segment_num <= get_max_segment_num(); ++segment_num)
    {
      s << '(' << min_ring_diff[segment_num] << ',' << max_ring_diff[segment_num] << ')';
    }
  s << std::endl;
  return s.str();
}

END_NAMESPACE_STIR
