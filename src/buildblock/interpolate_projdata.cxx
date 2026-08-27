//
//
/*
  Copyright (C) 2005 - 2009-10-27, Hammersmith Imanet Ltd
  Copyright (C) 2011-07-01 - 2011, Kris Thielemans
  Copyright 2023, Positrigo AG, Zurich
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
*/
/*!
 \file
 \ingroup projdata
 \brief Perform B-Splines Interpolation of sinograms

 \author Charalampos Tsoumpas
 \author Kris Thielemans
 \author Markus Jehl
*/
#include "stir/ProjData.h"
//#include "stir/display.h"
#include "stir/ProjDataInMemory.h"
#include "stir/ProjDataInterfile.h"
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoGenericNoArcCorr.h"
#include "stir/IndexRange.h"
#include "stir/BasicCoordinate.h"
#include "stir/Sinogram.h"
#include "stir/SegmentBySinogram.h"
#include "stir/Succeeded.h"
#include "stir/numerics/BSplines.h"
#include "stir/numerics/BSplinesRegularGrid.h"
#include "stir/interpolate_projdata.h"
#include "stir/extend_projdata.h"
#include "stir/numerics/sampling_functions.h"
#include "stir/error.h"
#include "stir/Coordinate4D.h"
#include "stir/IndexRange4D.h"
#include "stir/LORCoordinates.h"

#include <typeinfo>
#include <boost/format.hpp>
START_NAMESPACE_STIR

namespace detail_interpolate_projdata
{
/* Collection of functions to remove interleaving in non-arccorrected data.
It does this by doubling the number of views, and filling in the new
tangential positions by averaging the 4 neighbouring bins.
WARNING: most of STIR will get confused by the resulting sinograms,
so only use them here for the interpolate_projdata implementation.
*/

static shared_ptr<ProjDataInfo>
make_non_interleaved_proj_data_info(const ProjDataInfo& proj_data_info)
{

  if (dynamic_cast<ProjDataInfoCylindricalNoArcCorr const*>(&proj_data_info) == NULL)
    error("make_non_interleaved_proj_data is only appropriate for non-arccorrected data");

  shared_ptr<ProjDataInfo> new_proj_data_info_sptr(proj_data_info.clone());
  new_proj_data_info_sptr->set_num_views(proj_data_info.get_num_views() * 2);
  return new_proj_data_info_sptr;
}

// access Sinogram element with wrap-around and boundary conditions
static float
sino_element(const Sinogram<float>& sinogram, const int view_num, const int tangential_pos_num)
{
  assert(sinogram.get_min_view_num() == 0);
  const int num_views = sinogram.get_num_views();
  const int tang_pos_num = (view_num >= num_views ? -1 : 1) * tangential_pos_num;
  if (tang_pos_num < sinogram.get_min_tangential_pos_num() || tang_pos_num > sinogram.get_max_tangential_pos_num())
    return 0.F;
  else
    return sinogram[view_num % num_views][tang_pos_num];
}

static void
make_non_interleaved_sinogram(Sinogram<float>& out_sinogram, const Sinogram<float>& in_sinogram)
{
  if (is_null_ptr(dynamic_pointer_cast<const ProjDataInfoCylindricalNoArcCorr>(in_sinogram.get_proj_data_info_sptr())))
    error("make_non_interleaved_proj_data is only appropriate for non-arccorrected data");

  assert(out_sinogram.get_min_view_num() == 0);
  assert(in_sinogram.get_min_view_num() == 0);
  assert(out_sinogram.get_num_views() == in_sinogram.get_num_views() * 2);
  assert(in_sinogram.get_segment_num() == 0);
  assert(out_sinogram.get_segment_num() == 0);

  for (int view_num = out_sinogram.get_min_view_num(); view_num <= out_sinogram.get_max_view_num(); ++view_num)
    {
      for (int tangential_pos_num = out_sinogram.get_min_tangential_pos_num() + 1;
           tangential_pos_num <= out_sinogram.get_max_tangential_pos_num() - 1;
           ++tangential_pos_num)
        {
          if ((view_num + tangential_pos_num) % 2 == 0)
            {
              const int in_view_num = view_num % 2 == 0 ? view_num / 2 : (view_num + 1) / 2;
              out_sinogram[view_num][tangential_pos_num] = sino_element(in_sinogram, in_view_num, tangential_pos_num);
            }
          else
            {
              const int next_in_view = view_num / 2 + 1;
              const int other_in_view = (view_num + 1) / 2;

              out_sinogram[view_num][tangential_pos_num] = (sino_element(in_sinogram, view_num / 2, tangential_pos_num)
                                                            + sino_element(in_sinogram, next_in_view, tangential_pos_num)
                                                            + sino_element(in_sinogram, other_in_view, tangential_pos_num - 1)
                                                            + sino_element(in_sinogram, other_in_view, tangential_pos_num + 1))
                                                           / 4;
            }
        }
    }
}

#if 0
  // not needed for now
  static Sinogram<float>
  make_non_interleaved_sinogram(const ProjDataInfo& non_interleaved_proj_data_info,
                                const Sinogram<float>& in_sinogram)
  {
    Sinogram<float> out_sinogram =
      non_interleaved_proj_data_info.get_empty_sinogram(in_sinogram.get_axial_pos_num(),
                                                        in_sinogram.get_segment_num());

    make_non_interleaved_sinogram(out_sinogram, in_sinogram);
    return out_sinogram;
  }
#endif

static void
make_non_interleaved_segment(SegmentBySinogram<float>& out_segment, const SegmentBySinogram<float>& in_segment)
{
  if (is_null_ptr(dynamic_pointer_cast<const ProjDataInfoCylindricalNoArcCorr>(in_segment.get_proj_data_info_sptr())))
    error("make_non_interleaved_proj_data is only appropriate for non-arccorrected data");

  for (int axial_pos_num = out_segment.get_min_axial_pos_num(); axial_pos_num <= out_segment.get_max_axial_pos_num();
       ++axial_pos_num)
    {
      Sinogram<float> out_sinogram = out_segment.get_sinogram(axial_pos_num);
      make_non_interleaved_sinogram(out_sinogram, in_segment.get_sinogram(axial_pos_num));
      out_segment.set_sinogram(out_sinogram);
    }
}

static SegmentBySinogram<float>
make_non_interleaved_segment(const ProjDataInfo& non_interleaved_proj_data_info, const SegmentBySinogram<float>& in_segment)
{
  SegmentBySinogram<float> out_segment = non_interleaved_proj_data_info.get_empty_segment_by_sinogram(
      in_segment.get_segment_num(), in_segment.get_timing_pos_num());

  make_non_interleaved_segment(out_segment, in_segment);
  return out_segment;
}

} // end namespace detail_interpolate_projdata

using namespace detail_interpolate_projdata;

Succeeded
interpolate_projdata(ProjData& proj_data_out,
                     const ProjData& proj_data_in,
                     const BSpline::BSplineType these_types,
                     const bool remove_interleaving)
{
  BasicCoordinate<3, BSpline::BSplineType> these_types_3;
  these_types_3[1] = these_types_3[2] = these_types_3[3] = these_types;
  interpolate_projdata(proj_data_out, proj_data_in, these_types_3, remove_interleaving);
  return Succeeded::yes;
}

Succeeded
interpolate_projdata(ProjData& proj_data_out,
                     const ProjData& proj_data_in,
                     const BasicCoordinate<3, BSpline::BSplineType>& these_types,
                     const bool remove_interleaving)
{
  const ProjDataInfo& proj_data_in_info = *proj_data_in.get_proj_data_info_sptr();
  const ProjDataInfo& proj_data_out_info = *proj_data_out.get_proj_data_info_sptr();

  if (typeid(proj_data_in_info) != typeid(proj_data_out_info))
    {
      error("interpolate_projdata needs both projection data  to be of the same type\n"
            "(e.g. both arc-corrected or both not arc-corrected)");
    }

  // check for the same ring radius
  // This is strictly speaking only necessary for non-arccorrected data, but
  // we leave it in for all cases.
  if (fabs(proj_data_in_info.get_scanner_ptr()->get_inner_ring_radius()
           - proj_data_out_info.get_scanner_ptr()->get_inner_ring_radius())
      > 1)
    {
      error("interpolate_projdata needs both projection to be of a scanner with the same ring radius");
    }

  // handle BlocksOnCylindrical interpolation manually
  if (proj_data_in_info.get_scanner_sptr()->get_scanner_geometry() != "Cylindrical")
    return interpolate_blocks_on_cylindrical_projdata(proj_data_out, proj_data_in, remove_interleaving);

  // initialise interpolator
  BSpline::BSplinesRegularGrid<3, float, float> proj_data_interpolator(these_types);
  for (int k = proj_data_out_info.get_min_tof_pos_num(); k <= proj_data_out_info.get_max_tof_pos_num(); ++k)
    {
      SegmentBySinogram<float> segment
          = remove_interleaving ? make_non_interleaved_segment(*(make_non_interleaved_proj_data_info(proj_data_in_info)),
                                                               proj_data_in.get_segment_by_sinogram(0, k))
                                : proj_data_in.get_segment_by_sinogram(0, k);

      // for Cylindrical, spacing is regular in all directions, which makes mapping trivial
      std::function<BasicCoordinate<3, double>(const BasicCoordinate<3, int>&)> index_converter;
      // especially in view direction, extending by 5 leads to much smaller artifacts
      proj_data_interpolator.set_coef(extend_segment(segment, 5, 5, 5));

      BasicCoordinate<3, double> offset, step;
      // out_index * step + offset = in_index
      const float in_sampling_m = proj_data_in_info.get_sampling_in_m(Bin(0, 0, 0, 0));
      const float out_sampling_m = proj_data_out_info.get_sampling_in_m(Bin(0, 0, 0, 0));
      // offset in 'in' index units
      offset[1] = (proj_data_out_info.get_m(Bin(0, 0, 0, 0)) - proj_data_in_info.get_m(Bin(0, 0, 0, 0))) / in_sampling_m;
      step[1] = out_sampling_m / in_sampling_m;

      const float in_sampling_phi = (proj_data_in_info.get_phi(Bin(0, 1, 0, 0)) - proj_data_in_info.get_phi(Bin(0, 0, 0, 0)))
                                    / (remove_interleaving ? 2 : 1);
      const float out_sampling_phi = proj_data_out_info.get_phi(Bin(0, 1, 0, 0)) - proj_data_out_info.get_phi(Bin(0, 0, 0, 0));
      offset[2] = (proj_data_out_info.get_phi(Bin(0, 0, 0, 0)) - proj_data_in_info.get_phi(Bin(0, 0, 0, 0))) / in_sampling_phi;
      step[2] = out_sampling_phi / in_sampling_phi;

      const float in_sampling_s = proj_data_in_info.get_sampling_in_s(Bin(0, 0, 0, 0));
      const float out_sampling_s = proj_data_out_info.get_sampling_in_s(Bin(0, 0, 0, 0));
      offset[3] = (proj_data_out_info.get_s(Bin(0, 0, 0, 0)) - proj_data_in_info.get_s(Bin(0, 0, 0, 0))) / in_sampling_s;
      step[3] = out_sampling_s / in_sampling_s;

      // define a function to translate indices in the output proj data to indices in input proj data
      index_converter
          = [&proj_data_out_info, offset, step](const BasicCoordinate<3, int>& index_out) -> BasicCoordinate<3, double> {
        // translate to indices in input proj data
        BasicCoordinate<3, double> index_in;
        for (auto dim = 1; dim <= 3; dim++)
          index_in[dim] = index_out[dim] * step[dim] + offset[dim];

        return index_in;
      };

      SegmentBySinogram<float> sino_3D_out = proj_data_out.get_empty_segment_by_sinogram(0, false, k);
      sample_function_using_index_converter(sino_3D_out, proj_data_interpolator, index_converter);

      if (proj_data_out.set_segment(sino_3D_out) == Succeeded::no)
        return Succeeded::no;
    }
  return Succeeded::yes;
}

//! This function interpolates BlocksOnCylindrical proj data taking bucket intersections and gaps into account.
/*! The above proj data interpolation function works well for cylindrical scanners where there are no large discontinuities.
For BlocksOnCylindrical scanners, the sudden change of angle from one bucket to the next and the gaps between blocks and buckets
lead to interpolation artifacts if not taken into account. Therefore, this function does the interpolation in physical space
rather than directly on the proj data bin values. For each bin in proj_data_out (the full size proj data), we find the four
closest LORs in the downsampled proj_data_in. These are then weighted using bilinear interpolation based on the crystal positions
of the two endpoints of the LOR.
*/
Succeeded
interpolate_blocks_on_cylindrical_projdata(ProjData& proj_data_out, const ProjData& proj_data_in, bool remove_interleaving)
{
  const ProjDataInfo& proj_data_in_info = *proj_data_in.get_proj_data_info_sptr();
  const ProjDataInfo& proj_data_out_info = *proj_data_out.get_proj_data_info_sptr();

  auto m_offset = proj_data_in_info.get_m(Bin(0, 0, 0, 0));
  auto m_sampling = proj_data_in_info.get_sampling_in_m(Bin(0, 0, 0, 0));

  for (int k = proj_data_out_info.get_min_tof_pos_num(); k <= proj_data_out_info.get_max_tof_pos_num(); ++k)
    {
      SegmentBySinogram<float> segment
          = remove_interleaving ? make_non_interleaved_segment(*(make_non_interleaved_proj_data_info(proj_data_in_info)),
                                                               proj_data_in.get_segment_by_sinogram(0, k))
                                : proj_data_in.get_segment_by_sinogram(0, k);
      SegmentBySinogram<float> sino_3D_out = proj_data_out.get_empty_segment_by_sinogram(0, false, k);

      BasicCoordinate<3, int> min_out, max_out;
      IndexRange<3> out_range = sino_3D_out.get_index_range();
      if (!out_range.get_regular_range(min_out, max_out))
        warning("Output must be regular range!");

      // confirm that proj_data_in has equidistant sampling in m
      for (auto axial_pos = proj_data_in_info.get_min_axial_pos_num(0); axial_pos <= proj_data_in_info.get_max_axial_pos_num(0);
           axial_pos++)
        {
          if (abs(m_sampling - proj_data_in_info.get_sampling_in_m(Bin(0, 0, axial_pos, 0))) > 1E-4)
            error("input projdata to interpolate_projdata are not equidistantly sampled in m.");
        }

      BasicCoordinate<3, int> index_out;
      for (index_out[1] = min_out[1]; index_out[1] <= max_out[1]; ++index_out[1])
        {
          for (index_out[2] = min_out[2]; index_out[2] <= max_out[2]; ++index_out[2])
            {
              for (index_out[3] = min_out[3]; index_out[3] <= max_out[3]; ++index_out[3])
                {
                  // translate index on output to coordinate
                  auto bin
                = Bin(0 /* segment */, index_out[2] /* view */, 
                index_out[1] /* axial pos */, index_out[3] /* tangential pos
                */);
                  auto out_m = proj_data_out_info.get_m(bin);
                  double axial_idx = (out_m - m_offset) / m_sampling;
                  int axial_floor = std::floor(axial_idx);
                  int axial_ceil = std::ceil(axial_idx);
                  if (axial_floor == axial_ceil)
                    {
                      if (axial_floor == 0)
                        axial_ceil++;
                      else
                        axial_floor--;
                    }

                  // find the two crystals for this bin and on which buckets (modules) they are
                  int det1_num_out, det2_num_out;
                  const auto proj_data_out_info_ptr = dynamic_cast<const ProjDataInfoGenericNoArcCorr*>(&proj_data_out_info);
                  proj_data_out_info_ptr->get_det_num_pair_for_view_tangential_pos_num(det1_num_out,
                                                                                       det2_num_out,
                                                                                       index_out[2], /*view*/
                                                                                       index_out[3] /* tangential pos */);
                  const int dets_per_module_out = proj_data_out_info.get_scanner_sptr()->get_num_transaxial_crystals_per_bucket();
                  const int det1_module = det1_num_out / dets_per_module_out;
                  const int det2_module = det2_num_out / dets_per_module_out;

                  // translate the crystal position on each module from the full size scanner to the downsampled scanner
                  const auto proj_data_in_info_ptr = dynamic_cast<const ProjDataInfoGenericNoArcCorr*>(&proj_data_in_info);
                  const int dets_per_module_in
                      = proj_data_in_info_ptr->get_scanner_sptr()->get_num_transaxial_crystals_per_bucket();
                  const int crystal1_out_module_idx = det1_num_out % dets_per_module_out;
                  const double crystal1_out_module_pos
                      = std::floor(static_cast<double>(crystal1_out_module_idx)
                                   / proj_data_out_info_ptr->get_scanner_sptr()->get_num_transaxial_crystals_per_block())
                            * proj_data_out_info_ptr->get_scanner_sptr()->get_transaxial_block_spacing()
                        + static_cast<double>(
                              crystal1_out_module_idx
                              % proj_data_out_info_ptr->get_scanner_sptr()->get_num_transaxial_crystals_per_block())
                              * proj_data_out_info_ptr->get_scanner_sptr()->get_transaxial_crystal_spacing();
                  const double crystal1_num_in
                      = det1_module * dets_per_module_in
                        + crystal1_out_module_pos / proj_data_in_info.get_scanner_sptr()->get_transaxial_crystal_spacing();
                  const int crystal2_out_module_idx = det2_num_out % dets_per_module_out;
                  const double crystal2_out_module_pos
                      = std::floor(static_cast<double>(crystal2_out_module_idx)
                                   / proj_data_out_info_ptr->get_scanner_sptr()->get_num_transaxial_crystals_per_block())
                            * proj_data_out_info_ptr->get_scanner_sptr()->get_transaxial_block_spacing()
                        + static_cast<double>(
                              crystal2_out_module_idx
                              % proj_data_out_info_ptr->get_scanner_sptr()->get_num_transaxial_crystals_per_block())
                              * proj_data_out_info_ptr->get_scanner_sptr()->get_transaxial_crystal_spacing();
                  const double crystal2_num_in
                      = det2_module * dets_per_module_in
                        + crystal2_out_module_pos / proj_data_in_info.get_scanner_sptr()->get_transaxial_crystal_spacing();
                  const auto crystal1_num_in_floor
                      = std::max(static_cast<int>(std::floor(crystal1_num_in)), det1_module * dets_per_module_in);
                  const auto crystal1_num_in_ceil
                      = std::min(static_cast<int>(std::ceil(crystal1_num_in)), (det1_module + 1) * dets_per_module_in - 1);
                  const auto crystal2_num_in_floor
                      = std::max(static_cast<int>(std::floor(crystal2_num_in)), det2_module * dets_per_module_in);
                  const auto crystal2_num_in_ceil
                      = std::min(static_cast<int>(std::ceil(crystal2_num_in)), (det2_module + 1) * dets_per_module_in - 1);

                  // translate to indices in input proj data
                  BasicCoordinate<3, int> index_in;

                  // in this case we can skip parts of the interpolation
                  if (crystal1_num_in_floor == crystal1_num_in_ceil)
                    {
                      if (crystal2_num_in_floor == crystal2_num_in_ceil)
                        {
                          int one_view, one_tang;
                          proj_data_in_info_ptr->get_view_tangential_pos_num_for_det_num_pair(
                              one_view, one_tang, crystal1_num_in_floor, crystal2_num_in_floor);
                          one_tang = std::min(std::max(proj_data_in_info_ptr->get_min_tangential_pos_num(), one_tang),
                                              proj_data_in_info_ptr->get_max_tangential_pos_num());

                          index_in[1] = std::max(axial_floor, proj_data_in_info_ptr->get_min_axial_pos_num(0));
                          index_in[2] = one_view;
                          index_in[3] = one_tang;
                          sino_3D_out[index_out] += segment[index_in] * (axial_ceil - axial_idx);

                          index_in[1] = std::min(axial_ceil, proj_data_in_info_ptr->get_max_axial_pos_num(0));
                          sino_3D_out[index_out] += segment[index_in] * (axial_idx - axial_floor);
                        }
                      else
                        {
                          int ff_view, fc_view;
                          int ff_tang, fc_tang;
                          proj_data_in_info_ptr->get_view_tangential_pos_num_for_det_num_pair(
                              ff_view, ff_tang, crystal1_num_in_floor, crystal2_num_in_floor);
                          proj_data_in_info_ptr->get_view_tangential_pos_num_for_det_num_pair(
                              fc_view, fc_tang, crystal1_num_in_floor, crystal2_num_in_ceil);

                          ff_tang = std::min(std::max(proj_data_in_info_ptr->get_min_tangential_pos_num(), ff_tang),
                                             proj_data_in_info_ptr->get_max_tangential_pos_num());
                          fc_tang = std::min(std::max(proj_data_in_info_ptr->get_min_tangential_pos_num(), fc_tang),
                                             proj_data_in_info_ptr->get_max_tangential_pos_num());

                          index_in[1] = std::max(axial_floor, proj_data_in_info_ptr->get_min_axial_pos_num(0));
                          index_in[2] = ff_view;
                          index_in[3] = ff_tang;
                          sino_3D_out[index_out]
                              += segment[index_in] * (axial_ceil - axial_idx) * (crystal2_num_in_ceil - crystal2_num_in);
                          index_in[2] = fc_view;
                          index_in[3] = fc_tang;
                          sino_3D_out[index_out]
                              += segment[index_in] * (axial_ceil - axial_idx) * (crystal2_num_in - crystal2_num_in_floor);
                          index_in[1] = std::min(axial_ceil, proj_data_in_info_ptr->get_max_axial_pos_num(0));
                          sino_3D_out[index_out]
                              += segment[index_in] * (axial_idx - axial_floor) * (crystal2_num_in - crystal2_num_in_floor);
                          index_in[2] = ff_view;
                          index_in[3] = ff_tang;
                          sino_3D_out[index_out]
                              += segment[index_in] * (axial_idx - axial_floor) * (crystal2_num_in_ceil - crystal2_num_in);
                        }
                    }
                  else if (crystal2_num_in_floor == crystal2_num_in_ceil)
                    {
                      int ff_view, cf_view;
                      int ff_tang, cf_tang;
                      proj_data_in_info_ptr->get_view_tangential_pos_num_for_det_num_pair(
                          ff_view, ff_tang, crystal1_num_in_floor, crystal2_num_in_floor);
                      proj_data_in_info_ptr->get_view_tangential_pos_num_for_det_num_pair(
                          cf_view, cf_tang, crystal1_num_in_ceil, crystal2_num_in_floor);

                      ff_tang = std::min(std::max(proj_data_in_info_ptr->get_min_tangential_pos_num(), ff_tang),
                                         proj_data_in_info_ptr->get_max_tangential_pos_num());
                      cf_tang = std::min(std::max(proj_data_in_info_ptr->get_min_tangential_pos_num(), cf_tang),
                                         proj_data_in_info_ptr->get_max_tangential_pos_num());

                      index_in[1] = std::max(axial_floor, proj_data_in_info_ptr->get_min_axial_pos_num(0));
                      index_in[2] = ff_view;
                      index_in[3] = ff_tang;
                      sino_3D_out[index_out]
                          += segment[index_in] * (axial_ceil - axial_idx) * (crystal1_num_in_ceil - crystal1_num_in);
                      index_in[2] = cf_view;
                      index_in[3] = cf_tang;
                      sino_3D_out[index_out]
                          += segment[index_in] * (axial_ceil - axial_idx) * (crystal1_num_in - crystal1_num_in_floor);
                      index_in[1] = std::min(axial_ceil, proj_data_in_info_ptr->get_max_axial_pos_num(0));
                      sino_3D_out[index_out]
                          += segment[index_in] * (axial_idx - axial_floor) * (crystal1_num_in - crystal1_num_in_floor);
                      index_in[2] = ff_view;
                      index_in[3] = ff_tang;
                      sino_3D_out[index_out]
                          += segment[index_in] * (axial_idx - axial_floor) * (crystal1_num_in_ceil - crystal1_num_in);
                    }
                  else // in this case we need to do a trilinear interpolation
                    {
                      int ff_view, fc_view, cf_view, cc_view;
                      int ff_tang, fc_tang, cf_tang, cc_tang;
                      proj_data_in_info_ptr->get_view_tangential_pos_num_for_det_num_pair(
                          ff_view, ff_tang, crystal1_num_in_floor, crystal2_num_in_floor);
                      proj_data_in_info_ptr->get_view_tangential_pos_num_for_det_num_pair(
                          fc_view, fc_tang, crystal1_num_in_floor, crystal2_num_in_ceil);
                      proj_data_in_info_ptr->get_view_tangential_pos_num_for_det_num_pair(
                          cf_view, cf_tang, crystal1_num_in_ceil, crystal2_num_in_floor);
                      proj_data_in_info_ptr->get_view_tangential_pos_num_for_det_num_pair(
                          cc_view, cc_tang, crystal1_num_in_ceil, crystal2_num_in_ceil);

                      // TODO: why can we get positions out that are not even in the proj data?!
                      ff_tang = std::min(std::max(proj_data_in_info_ptr->get_min_tangential_pos_num(), ff_tang),
                                         proj_data_in_info_ptr->get_max_tangential_pos_num());
                      fc_tang = std::min(std::max(proj_data_in_info_ptr->get_min_tangential_pos_num(), fc_tang),
                                         proj_data_in_info_ptr->get_max_tangential_pos_num());
                      cf_tang = std::min(std::max(proj_data_in_info_ptr->get_min_tangential_pos_num(), cf_tang),
                                         proj_data_in_info_ptr->get_max_tangential_pos_num());
                      cc_tang = std::min(std::max(proj_data_in_info_ptr->get_min_tangential_pos_num(), cc_tang),
                                         proj_data_in_info_ptr->get_max_tangential_pos_num());

                      index_in[1] = std::max(axial_floor, proj_data_in_info_ptr->get_min_axial_pos_num(0));
                      index_in[2] = ff_view;
                      index_in[3] = ff_tang;
                      sino_3D_out[index_out] += segment[index_in] * (axial_ceil - axial_idx)
                                                * (crystal1_num_in_ceil - crystal1_num_in)
                                                * (crystal2_num_in_ceil - crystal2_num_in);
                      index_in[2] = fc_view;
                      index_in[3] = fc_tang;
                      sino_3D_out[index_out] += segment[index_in] * (axial_ceil - axial_idx)
                                                * (crystal1_num_in_ceil - crystal1_num_in)
                                                * (crystal2_num_in - crystal2_num_in_floor);
                      index_in[2] = cc_view;
                      index_in[3] = cc_tang;
                      sino_3D_out[index_out] += segment[index_in] * (axial_ceil - axial_idx)
                                                * (crystal1_num_in - crystal1_num_in_floor)
                                                * (crystal2_num_in - crystal2_num_in_floor);
                      index_in[2] = cf_view;
                      index_in[3] = cf_tang;
                      sino_3D_out[index_out] += segment[index_in] * (axial_ceil - axial_idx)
                                                * (crystal1_num_in - crystal1_num_in_floor)
                                                * (crystal2_num_in_ceil - crystal2_num_in);

                      index_in[1] = std::min(axial_ceil, proj_data_in_info_ptr->get_max_axial_pos_num(0));
                      index_in[2] = ff_view;
                      index_in[3] = ff_tang;
                      sino_3D_out[index_out] += segment[index_in] * (axial_idx - axial_floor)
                                                * (crystal1_num_in_ceil - crystal1_num_in)
                                                * (crystal2_num_in_ceil - crystal2_num_in);
                      index_in[2] = fc_view;
                      index_in[3] = fc_tang;
                      sino_3D_out[index_out] += segment[index_in] * (axial_idx - axial_floor)
                                                * (crystal1_num_in_ceil - crystal1_num_in)
                                                * (crystal2_num_in - crystal2_num_in_floor);
                      index_in[2] = cc_view;
                      index_in[3] = cc_tang;
                      sino_3D_out[index_out] += segment[index_in] * (axial_idx - axial_floor)
                                                * (crystal1_num_in - crystal1_num_in_floor)
                                                * (crystal2_num_in - crystal2_num_in_floor);
                      index_in[2] = cf_view;
                      index_in[3] = cf_tang;
                      sino_3D_out[index_out] += segment[index_in] * (axial_idx - axial_floor)
                                                * (crystal1_num_in - crystal1_num_in_floor)
                                                * (crystal2_num_in_ceil - crystal2_num_in);
                    }
                }
            }
        }
      if (proj_data_out.set_segment(sino_3D_out) == Succeeded::no)
        return Succeeded::no;
    }
  return Succeeded::yes;
}

Succeeded
interpolate_projdata_3d(ProjData& proj_data_out,
                        const ProjData& proj_data_in,
                        const BSpline::BSplineType& spline_type,
                        const bool use_view_offset)
{
  //   if (use_view_offset)
  //     warning("interpolate_projdata with use_view_offset is EXPERIMENTAL and NOT TESTED.");

  //   const shared_ptr<const ProjDataInfo> projdata_in_info_sptr = projdata_in_sptr->get_proj_data_info_sptr();

  //   shared_ptr<ProjDataInfo> projdata_in_up_info_sptr(projdata_in_sptr->get_proj_data_info_sptr()->clone());
  //   projdata_in_up_info_sptr->set_num_views(projdata_out->get_num_views());
  //   projdata_in_up_info_sptr->set_num_tangential_poss(projdata_out->get_num_tangential_poss());

  //   if (projdata_in_info_sptr->get_scanner_sptr()->get_scanner_geometry() == "BlocksOnCylindrical")
  //     {
  //       // interpolate_axial_position(*projdata_out, *projdata_in_sptr);
  //       return Succeeded::yes;
  //     }

  //   // check for the same ring radius
  //   // This is strictly speaking only necessary for non-arccorrected data, but
  //   // we leave it in for all cases.
  //   if (fabs(projdata_in_info_sptr->get_scanner_sptr()->get_inner_ring_radius()
  //            - projdata_out->get_proj_data_info_sptr()->get_scanner_sptr()->get_inner_ring_radius())
  //       > 1)
  //     {
  //       error("interpolate_projdata needs both projection to be of a scanner with the same ring radius");
  //     }

  //   std::string output_filename = "tmp_in_up";
  //   //    ProjDataInMemory in_up_projdata(projdata_in_sptr->get_exam_info_sptr(),
  //   //                                    projdata_in_info_sptr->create_shared_clone(),
  //   //                                    1); // I pressume 1 but we should check!
  //   // For larger scanners ProjDataInterfile might be the right option.
  //   ProjDataInterfile in_up_projdata(
  //       projdata_in_sptr->get_exam_info_sptr(), projdata_in_up_info_sptr, output_filename, std::ios::out);

  //   info("interpolate_projdata: Interpolating views and tangential positions ...");

  //   bool flag_something_went_wrong = false;

  // #ifdef STIR_OPENMP
  // #  if _OPENMP < 201107
  // #    pragma omp parallel for
  // #  else
  // #    pragma omp parallel for schedule(dynamic)
  // #  endif
  // #endif
  // #ifdef STIR_TOF
  //   for (int i_tof_in = projdata_in_sptr->get_min_tof_pos_num(); i_tof_in <= projdata_in_sptr->get_max_tof_pos_num();
  //   ++i_tof_in)
  //     {
  // #endif
  //       for (int i_seg_in = projdata_in_sptr->get_min_segment_num(); i_seg_in <= projdata_in_sptr->get_max_segment_num();
  //            ++i_seg_in)
  //         {
  //           info(boost::format("Now processing segment #: %1%") % i_seg_in);

  //           BasicCoordinate<2, BSpline::BSplineType> these_types2;
  //           these_types2[1] = these_types2[2] = spline_type;
  //           BasicCoordinate<2, double> offset, step;

  //           const float in_sampling_phi = (projdata_in_info_sptr->get_phi(Bin(i_seg_in, 1, 0, 0 /*, i_tof_in*/))
  //                                          - projdata_in_info_sptr->get_phi(Bin(i_seg_in, 0, 0, 0 /*, i_tof_in*/)));
  //           const float in_up_sampling_phi
  //               = projdata_out->get_proj_data_info_sptr()->get_phi(Bin(i_seg_in, 1, 0, 0 /*, i_tof_in*/))
  //                 - projdata_out->get_proj_data_info_sptr()->get_phi(Bin(i_seg_in, 0, 0, 0 /*, i_tof_in*/));

  //           const float in_view_offset = 0;
  //           //                    use_view_offset
  //           //                    ? projdata_in_info_sptr->get_scanner_ptr()->get_intrinsic_azimuthal_tilt()
  //           //                    : 0.F;
  //           const float in_up_view_offset = 0;
  //           //                    use_view_offset
  //           //                    ? projdata_in_up_info_sptr->get_scanner_ptr()->get_intrinsic_azimuthal_tilt()
  //           //                    : 0.F;

  //           offset[1]
  //               = (projdata_in_info_sptr->get_phi(Bin(i_seg_in, 0, 0, 0 /*, i_tof_in*/)) + in_view_offset
  //                  - projdata_out->get_proj_data_info_sptr()->get_phi(Bin(i_seg_in, 0, 0, 0 /*, i_tof_in*/)) -
  //                  in_up_view_offset)
  //                 / in_sampling_phi;
  //           step[1] = in_up_sampling_phi / in_sampling_phi;

  //           const float in_sampling_s = projdata_in_info_sptr->get_sampling_in_s(Bin(i_seg_in, 0, 0, 0 /*, i_tof_in*/));
  //           const float in_up_sampling_s
  //               = projdata_out->get_proj_data_info_sptr()->get_sampling_in_s(Bin(i_seg_in, 0, 0, 0 /*, i_tof_in*/));

  //           float dd = projdata_out->get_proj_data_info_sptr()->get_s(Bin(i_seg_in, 0, 0, 0 /*, i_tof_in*/));
  //           float ee = projdata_in_info_sptr->get_s(Bin(i_seg_in, 0, 0, 0 /*, i_tof_in*/));

  //           offset[2] = (dd - ee) / in_sampling_s;
  //           step[2] = in_up_sampling_s / in_sampling_s;

  //           std::cout << "n3" << std::endl;
  //           int inv_seg = -i_seg_in;

  //           Array<3, float> extended3 = extend_segment_in_views(projdata_in_sptr->get_segment_by_sinogram(i_seg_in /*,
  //           i_tof_in*/),
  //                                                               2,
  //                                                               2,
  //                                                               projdata_in_sptr->get_segment_by_sinogram(inv_seg /*,
  //                                                               i_tof_in*/));
  //           SegmentBySinogram<float> sino_3D_out
  //               = in_up_projdata.get_proj_data_info_sptr()->get_empty_segment_by_sinogram(i_seg_in);

  //           std::cout << "n6" << std::endl;
  //           for (int i_axial = projdata_in_sptr->get_min_axial_pos_num(i_seg_in);
  //                i_axial <= projdata_in_sptr->get_max_axial_pos_num(i_seg_in);
  //                ++i_axial)
  //             {
  //               Array<2, float> extended = extended3[i_axial];
  //               BSpline::BSplinesRegularGrid<2, float, float> proj_data_interpolator(these_types2);
  //               for (int y = extended.get_min_index(); y <= extended.get_max_index(); ++y)
  //                 {
  //                   const int old_min = extended[y].get_min_index();
  //                   const int old_max = extended[y].get_max_index();
  //                   extended[y].grow(old_min - 1, old_max + 1);
  //                   extended[y][old_min - 1] = extended[y][old_min];
  //                   extended[y][old_max + 1] = extended[y][old_max];
  //                 }
  //               std::cout << "n7" << std::endl;
  //               proj_data_interpolator.set_coef(extended);
  //               std::cout << "n7.2" << std::endl;
  //               Sinogram<float> sino_2D_out = sino_3D_out.get_sinogram(i_axial);
  //               sample_function_on_regular_grid(sino_2D_out, proj_data_interpolator, offset, step);
  //               sino_3D_out.set_sinogram(sino_2D_out, i_axial);
  //             }
  //           if (in_up_projdata.set_segment(sino_3D_out) == Succeeded::no)
  //             {
  //               flag_something_went_wrong = true;
  //             }
  //         }

  // #ifdef STIR_TOF
  //     }
  // #endif

  //   if (flag_something_went_wrong == true)
  //     error("interpolate_projdata: Something went wrong in the first level of interpolation");

  //   info("interpolate_projdata: Finished interpolating views and tangential positions!");

  //   const stir::shared_ptr<ProjData> in_up_projdata_in = ProjData::read_from_file("tmp_in_up.hs");
  //   //    std::cout << in_up_projdata_in->get_proj_data_info_sptr()->parameter_info() << std::endl;

  //   //    std::cout << projdata_out->get_proj_data_info_sptr()->parameter_info() << std::endl;

  //   //    if(projdata_out->get_num_segments() == 1)
  //   //    {
  //   //        SegmentByView<float> v = in_up_projdata_in->get_segment_by_view(0);
  //   //        projdata_out->set_segment(v);
  //   //        int summm = v.sum();

  //   //        std::cout << "in_up_projdata_in sum: " << summm << std::endl;
  //   //        summm= projdata_out->get_segment_by_sinogram(0).sum();
  //   //        std::cout << "projdata_out sum: " << summm << std::endl;
  //   //        return Succeeded::yes;
  //   //    }

  //   const shared_ptr<const ProjDataInfoCylindricalNoArcCorr> in_up_projdata_in_info_sptr
  //       = dynamic_pointer_cast<ProjDataInfoCylindricalNoArcCorr>(
  //           in_up_projdata_in->get_proj_data_info_sptr()->create_shared_clone());

  //   if (is_null_ptr(in_up_projdata_in_info_sptr))
  //     error("BlockGeometry or arc correction, whye are we here??");

  //   // Compressed data are not supported
  //   if (in_up_projdata_in_info_sptr->get_min_ring_difference(0) != in_up_projdata_in_info_sptr->get_max_ring_difference(0))
  //     {
  //       warning("Compressed data are not supported, yet!");
  //       return Succeeded::no;
  //     }

  //   const shared_ptr<const ProjDataInfoCylindricalNoArcCorr> out_projdata_info_sptr
  //       =
  //       dynamic_pointer_cast<ProjDataInfoCylindricalNoArcCorr>(projdata_out->get_proj_data_info_sptr()->create_shared_clone());

  //   if (is_null_ptr(out_projdata_info_sptr))
  //     error("Interpolate_projdata: Interpolation in 3D needs non arc corrected data.");

  //   //    const stir::shared_ptr<const ProjDataInfo> projdata_out_info = projdata_out->get_proj_data_info_sptr();

  //   // Add two more rings, to have a wider space for the interpolation
  //   IndexRange4D mich_index(0,
  //                           in_up_projdata_in_info_sptr->get_scanner_sptr()->get_num_rings() + 1,
  //                           0,
  //                           in_up_projdata_in_info_sptr->get_scanner_sptr()->get_num_rings() + 1,
  //                           0,
  //                           out_projdata_info_sptr->get_scanner_sptr()->get_num_detectors_per_ring() - 1,
  //                           0,
  //                           out_projdata_info_sptr->get_scanner_sptr()->get_num_detectors_per_ring() - 1);

  //   BasicCoordinate<4, BSpline::BSplineType> these_types_4;
  //   these_types_4[1] = these_types_4[2] = these_types_4[3] = these_types_4[4] = spline_type;

  // #ifdef STIR_TOF // Now it is better to NOT parallelise the TOF bins but the geometric loops.
  //   info("interpolate_projdata: Creating michelogram for 3D interpolation ...");
  //   for (int i_tof_in = in_up_projdata_in->get_min_tof_pos_num(); i_tof_in <= in_up_projdata_in->get_max_tof_pos_num();
  //   ++i_tof_in)
  //     {
  //       const int cur_tof = i_tof_in;
  // #endif

  //       // Create the 4D Michelogram R1.R1.D1.D2
  //       Array<4, float> downsampled_array_4d(mich_index);
  //       downsampled_array_4d.fill(0.0);

  // #ifdef STIR_OPENMP
  // #  pragma omp parallel for schedule(dynamic) // collapse(3)
  // #endif
  //       for (int i_seg = in_up_projdata_in_info_sptr->get_min_segment_num();
  //            i_seg <= in_up_projdata_in_info_sptr->get_max_segment_num();
  //            ++i_seg)
  //         {
  //           int min_axial_pos = in_up_projdata_in_info_sptr->get_min_axial_pos_num(i_seg);
  //           int max_axial_pos = in_up_projdata_in_info_sptr->get_max_axial_pos_num(i_seg);

  //           //             info( std::string("interpolate_projdata: Processing segment:") << i_seg );
  //           std::cout << "interpolate_projdata: Processing segment:" << i_seg << std::endl;

  //           SegmentBySinogram<float> sino3D = in_up_projdata_in->get_segment_by_sinogram(i_seg /*, cur_tof*/);

  //           for (int i_axial = min_axial_pos; i_axial <= max_axial_pos; ++i_axial)
  //             {
  //               int r1, r2;
  //               in_up_projdata_in_info_sptr->get_ring_pair_for_segment_axial_pos_num(r1, r2, i_seg, i_axial);
  //               // For the view and tangential position we have to use the finer template, as otherwise we might run
  //               // into problems with the downsampled.
  //               for (int i_view = in_up_projdata_in_info_sptr->get_min_view_num();
  //                    i_view <= in_up_projdata_in_info_sptr->get_max_view_num();
  //                    ++i_view)
  //                 {
  //                   for (int i_tang = in_up_projdata_in_info_sptr->get_min_tangential_pos_num();
  //                        i_tang <= in_up_projdata_in_info_sptr->get_max_tangential_pos_num();
  //                        ++i_tang)
  //                     {
  //                       int c1, c2;
  //                       out_projdata_info_sptr->get_det_num_pair_for_view_tangential_pos_num(c1, c2, i_view, i_tang);
  //                       // dynamic_cast<ProjDataInfoCylindricalNoArcCorr*
  //                       // >(in_up_projdata.get_proj_data_info_sptr().get())->get_det_pair_for_bin(c1, r1, c2, r2, tmp_bin);
  //                       // std::cout << r2 << " " << r1 << " " << c2 << " " << c1 << std::endl;
  //                       downsampled_array_4d[r1 + 1][r2 + 1][c1][c2] = sino3D[i_axial][i_view][i_tang];
  //                     }
  //                 }
  //             }
  //         }
  //       std::cout << "kkd" << std::endl;
  //       // Fill the extra rings with the values from their neigbours
  //       int last_ring = in_up_projdata_in_info_sptr->get_scanner_sptr()->get_num_rings();
  //       for (int i_c1 = 0; i_c1 < out_projdata_info_sptr->get_scanner_sptr()->get_num_detectors_per_ring(); ++i_c1)
  //         {
  //           for (int i_c2 = 0; i_c2 < out_projdata_info_sptr->get_scanner_sptr()->get_num_detectors_per_ring(); ++i_c2)
  //             {
  //               downsampled_array_4d[0][0][i_c1][i_c2] = downsampled_array_4d[1][1][i_c1][i_c2];
  //               downsampled_array_4d[last_ring + 1][last_ring + 1][i_c1][i_c2]
  //                   = downsampled_array_4d[last_ring][last_ring][i_c1][i_c2];
  //             }
  //         }

  //       BSpline::BSplinesRegularGrid<4, float, float> mich_data_interpolator(spline_type);
  //       mich_data_interpolator.set_coef(downsampled_array_4d);

  //       info("interpolate_projdata: Finished michelogram!");

  //       const float angle_spacing = 1
  //                                   / (in_up_projdata_in_info_sptr->get_phi(Bin(0, 1, 0, 0 /*, cur_tof*/))
  //                                      - in_up_projdata_in_info_sptr->get_phi(Bin(0, 0, 0, 0 /*, cur_tof*/)));
  //       const float eff_ring_radius = out_projdata_info_sptr->get_scanner_sptr()->get_effective_ring_radius();

  // #ifdef STIR_OPENMP
  // #  pragma omp parallel for schedule(dynamic)
  // #endif
  //       for (int i_seg = out_projdata_info_sptr->get_min_segment_num(); i_seg <= out_projdata_info_sptr->get_max_segment_num();
  //            ++i_seg)
  //         {
  //           int min_axial_pos = out_projdata_info_sptr->get_min_axial_pos_num(i_seg);
  //           int max_axial_pos = out_projdata_info_sptr->get_max_axial_pos_num(i_seg);
  //           SegmentBySinogram<float> _sino3D = out_projdata_info_sptr->get_empty_segment_by_sinogram(i_seg, false /*,
  //           cur_tof*/);
  //           // info(boost::format("interpolate_projdata: Interpolating michelogram, segment: %1%") % i_seg);

  //           for (int i_axial = min_axial_pos; i_axial <= max_axial_pos; ++i_axial)
  //             {
  //               const double ring_spacing_in = 1 / in_up_projdata_in_info_sptr->get_ring_spacing();
  //               const double ring_spacing_out = out_projdata_info_sptr->get_ring_spacing();
  //               const float offset = 0.5 * in_up_projdata_in_info_sptr->get_ring_spacing()
  //                                    + out_projdata_info_sptr->get_scanner_sptr()->get_num_rings() * ring_spacing_out * 0.5;

  //               for (int i_view = out_projdata_info_sptr->get_min_view_num(); i_view <=
  //               out_projdata_info_sptr->get_max_view_num();
  //                    ++i_view)
  //                 {
  //                   for (int i_tang = out_projdata_info_sptr->get_min_tangential_pos_num();
  //                        i_tang <= out_projdata_info_sptr->get_max_tangential_pos_num();
  //                        ++i_tang)
  //                     {
  //                       // std::cout << i_axial << " " << i_view << " " << i_tang << std::endl;
  //                       float value = 0.0;
  //                       Bin tmp_bin(i_seg, i_view, i_axial, i_tang, /*cur_tof,*/ value);

  //                       LORInAxialAndNoArcCorrSinogramCoordinates<float> lor_sino;
  //                       LORInCylinderCoordinates<float> lor_cyl;

  //                       out_projdata_info_sptr->get_LOR(lor_sino, tmp_bin);
  //                       int d1, d2, dr1, dr2;
  //                       out_projdata_info_sptr->get_det_pair_for_bin(d1, dr1, d2, dr2, tmp_bin);
  //                       lor_sino.change_representation(lor_cyl, eff_ring_radius);

  //                       double c1(lor_cyl.p1().psi() * angle_spacing);
  //                       double c2(lor_cyl.p2().psi() * angle_spacing);
  //                       double r1((lor_cyl.p1().z() + offset) * (ring_spacing_in));
  //                       double r2((lor_cyl.p2().z() + offset) * (ring_spacing_in));

  //                       // int cur_tilt, cur_slice, cur_phi, cur_rad;
  //                       // sinoMap.getSinoCoordsForCrystals(
  //                       //		 0, d1, dr1,
  //                       //		 0, d2, dr2,
  //                       //		 &cur_tilt, &cur_slice, &cur_phi, &cur_rad, 0);

  //                       //         if(!is_gap(dr1) && !is_gap(dr2))
  //                       {
  //                         //                         std::cout << r1 << " " << r2 << " " << c1 << " " << c2 << std::endl;
  //                         BasicCoordinate<4, double> pos = make_coordinate(r1, r2, c1, c2);
  //                         // if(cur_phi * mh_normsino.numray + cur_rad > sliceSize ||
  //                         // cur_slice > norm_seg.size())
  //                         // std::cout << "Nikos: Overflow!" << cur_slice << " of " << norm_seg.size() << ", " <<
  //                         // cur_phi*mh_normsino.numray + cur_rad <<" of " << sliceSize << std::endl;
  //                         value = mich_data_interpolator(pos); // * norm_seg[cur_slice][cur_phi * mh_normsino.numray +
  //                         cur_rad];
  //                       }

  //                       _sino3D[i_axial][i_view][i_tang] = value;
  //                     }
  //                 }
  //             }
  //           projdata_out->set_segment(_sino3D);
  //         }

  // #ifdef STIR_TOF
  //     }
  // #endif

  info("interpolate_projdata: Finished michelogram interpolation!");
  return Succeeded::yes;
}

END_NAMESPACE_STIR
