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
#include <typeinfo>

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

  // initialise interpolator
  BSpline::BSplinesRegularGrid<3, float, float> proj_data_interpolator(these_types);
  for (int k = proj_data_out_info.get_min_tof_pos_num(); k <= proj_data_out_info.get_max_tof_pos_num(); ++k)
    {
      SegmentBySinogram<float> segment
          = remove_interleaving ? make_non_interleaved_segment(*(make_non_interleaved_proj_data_info(proj_data_in_info)),
                                                               proj_data_in.get_segment_by_sinogram(0, k))
                                : proj_data_in.get_segment_by_sinogram(0, k);

      std::function<BasicCoordinate<3, double>(const BasicCoordinate<3, int>&)> index_converter;
      if (proj_data_in_info.get_scanner_sptr()->get_scanner_geometry() == "Cylindrical")
        { // for Cylindrical, spacing is regular in all directions, which makes mapping trivial
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
          const float out_sampling_phi
              = proj_data_out_info.get_phi(Bin(0, 1, 0, 0)) - proj_data_out_info.get_phi(Bin(0, 0, 0, 0));
          offset[2]
              = (proj_data_out_info.get_phi(Bin(0, 0, 0, 0)) - proj_data_in_info.get_phi(Bin(0, 0, 0, 0))) / in_sampling_phi;
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
        }
      else
        { // for BlocksOnCylindrical, views and tangential positions are scaled by a fixed value
          auto transaxial_bucket_size_in = (proj_data_in_info.get_scanner_sptr()->get_num_transaxial_blocks_per_bucket() - 1)
                                               * proj_data_in_info.get_scanner_sptr()->get_transaxial_block_spacing()
                                           + (proj_data_in_info.get_scanner_sptr()->get_num_transaxial_crystals_per_block() - 1)
                                                 * proj_data_in_info.get_scanner_sptr()->get_transaxial_crystal_spacing();
          auto transaxial_bucket_size_out = (proj_data_out_info.get_scanner_sptr()->get_num_transaxial_blocks_per_bucket() - 1)
                                                * proj_data_out_info.get_scanner_sptr()->get_transaxial_block_spacing()
                                            + (proj_data_out_info.get_scanner_sptr()->get_num_transaxial_crystals_per_block() - 1)
                                                  * proj_data_out_info.get_scanner_sptr()->get_transaxial_crystal_spacing();
          // TODO: for now assuming the bucket sizes are the same (so ignoring the above and not adding an offset when translating
          // crystal positions)
          auto scale_factor = (double)proj_data_out_info.get_scanner_sptr()->get_transaxial_crystal_spacing()
                              / (double)proj_data_in_info.get_scanner_sptr()->get_transaxial_crystal_spacing();

          // only extending in axial direction - an extension of 2 was found to be sufficient
          proj_data_interpolator.set_coef(extend_segment(segment, 5, 2, 5));

          auto m_offset = proj_data_in_info.get_m(Bin(0, 0, 0, 0));
          auto m_sampling = proj_data_in_info.get_sampling_in_m(Bin(0, 0, 0, 0));

          // confirm that proj_data_in has equidistant sampling in m
          for (auto axial_pos = proj_data_in_info.get_min_axial_pos_num(0);
               axial_pos <= proj_data_in_info.get_max_axial_pos_num(0);
               axial_pos++)
            {
              if (abs(m_sampling - proj_data_in_info.get_sampling_in_m(Bin(0, 0, axial_pos, 0))) > 1E-4)
                error("input projdata to interpolate_projdata are not equidistantly sampled in m.");
            }

          // define a function to translate indices in the output proj data to indices in input proj data
          index_converter = [&proj_data_out_info, &proj_data_in_info, m_offset, m_sampling, scale_factor](
                                const BasicCoordinate<3, int>& index_out) -> BasicCoordinate<3, double> {
            // translate index on output to coordinate
            auto bin
                = Bin(0 /* segment */, index_out[2] /* view */, index_out[1] /* axial pos */, index_out[3] /* tangential pos
                */);
            auto out_m = proj_data_out_info.get_m(bin);

            // translate to indices in input proj data
            BasicCoordinate<3, double> index_in;
            index_in[1] = (out_m - m_offset) / m_sampling;

            int det1_num_out, det2_num_out;
            const auto proj_data_out_info_ptr = dynamic_cast<const ProjDataInfoGenericNoArcCorr*>(&proj_data_out_info);
            proj_data_out_info_ptr->get_det_num_pair_for_view_tangential_pos_num(det1_num_out,
                                                                                 det2_num_out,
                                                                                 index_out[2], /*view*/
                                                                                 index_out[3] /* tangential pos */);
            const int dets_per_module_out = proj_data_out_info.get_scanner_sptr()->get_num_transaxial_crystals_per_bucket();
            const int det1_module = std::floor(det1_num_out / dets_per_module_out);
            const int det2_module = std::floor(det2_num_out / dets_per_module_out);

            const auto proj_data_in_info_ptr = dynamic_cast<const ProjDataInfoGenericNoArcCorr*>(&proj_data_in_info);
            const int dets_per_module_in = proj_data_in_info_ptr->get_scanner_sptr()->get_num_transaxial_crystals_per_bucket();
            double crystal1_num_in
                = det1_module * dets_per_module_in + static_cast<double>(det1_num_out % dets_per_module_out) * scale_factor;
            double crystal2_num_in
                = det2_module * dets_per_module_in + static_cast<double>(det2_num_out % dets_per_module_out) * scale_factor;
            auto crystal1_num_in_floor
                = std::max(static_cast<int>(std::floor(crystal1_num_in)), det1_module * dets_per_module_in);
            auto crystal1_num_in_ceil
                = std::min(static_cast<int>(std::ceil(crystal1_num_in)), (det1_module + 1) * dets_per_module_in - 1);
            auto crystal2_num_in_floor
                = std::max(static_cast<int>(std::floor(crystal2_num_in)), det2_module * dets_per_module_in);
            auto crystal2_num_in_ceil
                = std::min(static_cast<int>(std::ceil(crystal2_num_in)), (det2_module + 1) * dets_per_module_in - 1);

            int ground_truth_view, ground_truth_tang;
            proj_data_in_info_ptr->get_view_tangential_pos_num_for_det_num_pair(ground_truth_view,
                                                                                ground_truth_tang,
                                                                                static_cast<int>(std::round(crystal1_num_in)),
                                                                                static_cast<int>(std::round(crystal2_num_in)));

            // in this case we can skip parts of the interpolation
            if (crystal1_num_in_floor == crystal1_num_in_ceil)
              {
                if (crystal2_num_in_floor == crystal2_num_in_ceil)
                  {
                    int one_view, one_tang;
                    proj_data_in_info_ptr->get_view_tangential_pos_num_for_det_num_pair(
                        one_view, one_tang, crystal1_num_in_floor, crystal2_num_in_floor);
                    index_in[2] = one_view;
                    index_in[3] = one_tang;
                  }
                else
                  {
                    int ff_view, fc_view;
                    int ff_tang, fc_tang;
                    proj_data_in_info_ptr->get_view_tangential_pos_num_for_det_num_pair(
                        ff_view, ff_tang, crystal1_num_in_floor, crystal2_num_in_floor);
                    proj_data_in_info_ptr->get_view_tangential_pos_num_for_det_num_pair(
                        fc_view, fc_tang, crystal1_num_in_floor, crystal2_num_in_ceil);

                    // check if one of the views or tangential positions is out of line
                    if (abs(ff_view - ground_truth_view) > proj_data_in_info_ptr->get_num_views() / 2)
                      {
                        if (ff_view < ground_truth_view)
                          ff_view += proj_data_in_info_ptr->get_num_views(); // extend at the end
                        else
                          ff_view -= proj_data_in_info_ptr->get_num_views(); // extend at the front
                        ff_tang = std::min(std::max(-ff_tang, proj_data_in_info_ptr->get_min_tangential_pos_num()),
                                           proj_data_in_info_ptr->get_max_tangential_pos_num());
                      }
                    if (abs(fc_view - ground_truth_view) > proj_data_in_info_ptr->get_num_views() / 2)
                      {
                        if (fc_view < ground_truth_view)
                          fc_view += proj_data_in_info_ptr->get_num_views(); // extend at the end
                        else
                          fc_view -= proj_data_in_info_ptr->get_num_views(); // extend at the front
                        fc_tang = std::min(std::max(-fc_tang, proj_data_in_info_ptr->get_min_tangential_pos_num()),
                                           proj_data_in_info_ptr->get_max_tangential_pos_num());
                      }

                    index_in[2] = ff_view * (crystal2_num_in_ceil - crystal2_num_in)
                                  + fc_view * (crystal2_num_in - crystal2_num_in_floor);
                    index_in[3] = ff_tang * (crystal2_num_in_ceil - crystal2_num_in)
                                  + fc_tang * (crystal2_num_in - crystal2_num_in_floor);
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

                // check if one of the views or tangential positions is out of line
                if (abs(ff_view - ground_truth_view) > proj_data_in_info_ptr->get_num_views() / 2)
                  {
                    if (ff_view < ground_truth_view)
                      ff_view += proj_data_in_info_ptr->get_num_views(); // extend at the end
                    else
                      ff_view -= proj_data_in_info_ptr->get_num_views(); // extend at the front
                    ff_tang = std::min(std::max(-ff_tang, proj_data_in_info_ptr->get_min_tangential_pos_num()),
                                       proj_data_in_info_ptr->get_max_tangential_pos_num());
                  }
                if (abs(cf_view - ground_truth_view) > proj_data_in_info_ptr->get_num_views() / 2)
                  {
                    if (cf_view < ground_truth_view)
                      cf_view += proj_data_in_info_ptr->get_num_views(); // extend at the end
                    else
                      cf_view -= proj_data_in_info_ptr->get_num_views(); // extend at the front
                    cf_tang = std::min(std::max(-cf_tang, proj_data_in_info_ptr->get_min_tangential_pos_num()),
                                       proj_data_in_info_ptr->get_max_tangential_pos_num());
                  }

                index_in[2]
                    = ff_view * (crystal1_num_in_ceil - crystal1_num_in) + cf_view * (crystal1_num_in - crystal1_num_in_floor);
                index_in[3]
                    = ff_tang * (crystal1_num_in_ceil - crystal1_num_in) + cf_tang * (crystal1_num_in - crystal1_num_in_floor);
              }
            else // in this case we need to do a bilinear interpolation
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

                // check if one of the views or tangential positions is out of line
                if (abs(ff_view - ground_truth_view) > proj_data_in_info_ptr->get_num_views() / 2)
                  {
                    if (ff_view < ground_truth_view)
                      ff_view += proj_data_in_info_ptr->get_num_views(); // extend at the end
                    else
                      ff_view -= proj_data_in_info_ptr->get_num_views(); // extend at the front
                    ff_tang = std::min(std::max(-ff_tang, proj_data_in_info_ptr->get_min_tangential_pos_num()),
                                       proj_data_in_info_ptr->get_max_tangential_pos_num());
                  }
                if (abs(fc_view - ground_truth_view) > proj_data_in_info_ptr->get_num_views() / 2)
                  {
                    if (fc_view < ground_truth_view)
                      fc_view += proj_data_in_info_ptr->get_num_views(); // extend at the end
                    else
                      fc_view -= proj_data_in_info_ptr->get_num_views(); // extend at the front
                    fc_tang = std::min(std::max(-fc_tang, proj_data_in_info_ptr->get_min_tangential_pos_num()),
                                       proj_data_in_info_ptr->get_max_tangential_pos_num());
                  }
                if (abs(cf_view - ground_truth_view) > proj_data_in_info_ptr->get_num_views() / 2)
                  {
                    if (cf_view < ground_truth_view)
                      cf_view += proj_data_in_info_ptr->get_num_views(); // extend at the end
                    else
                      cf_view -= proj_data_in_info_ptr->get_num_views(); // extend at the front
                    cf_tang = std::min(std::max(-cf_tang, proj_data_in_info_ptr->get_min_tangential_pos_num()),
                                       proj_data_in_info_ptr->get_max_tangential_pos_num());
                  }
                if (abs(cc_view - ground_truth_view) > proj_data_in_info_ptr->get_num_views() / 2)
                  {
                    if (cc_view < ground_truth_view)
                      cc_view += proj_data_in_info_ptr->get_num_views(); // extend at the end
                    else
                      cc_view -= proj_data_in_info_ptr->get_num_views(); // extend at the front
                    cc_tang = std::min(std::max(-cc_tang, proj_data_in_info_ptr->get_min_tangential_pos_num()),
                                       proj_data_in_info_ptr->get_max_tangential_pos_num());
                  }

                // for the next two, we need to do a bilinear interpolation
                index_in[2] = ff_view * (crystal1_num_in_ceil - crystal1_num_in) * (crystal2_num_in_ceil - crystal2_num_in)
                              + fc_view * (crystal1_num_in_ceil - crystal1_num_in) * (crystal2_num_in - crystal2_num_in_floor)
                              + cf_view * (crystal1_num_in - crystal1_num_in_floor) * (crystal2_num_in_ceil - crystal2_num_in)
                              + cc_view * (crystal1_num_in - crystal1_num_in_floor) * (crystal2_num_in - crystal2_num_in_floor);
                index_in[3] = ff_tang * (crystal1_num_in_ceil - crystal1_num_in) * (crystal2_num_in_ceil - crystal2_num_in)
                              + fc_tang * (crystal1_num_in_ceil - crystal1_num_in) * (crystal2_num_in - crystal2_num_in_floor)
                              + cf_tang * (crystal1_num_in - crystal1_num_in_floor) * (crystal2_num_in_ceil - crystal2_num_in)
                              + cc_tang * (crystal1_num_in - crystal1_num_in_floor) * (crystal2_num_in - crystal2_num_in_floor);
              }

            return index_in;
          };
        }

      SegmentBySinogram<float> sino_3D_out = proj_data_out.get_empty_segment_by_sinogram(0, false, k);
      sample_function_using_index_converter(sino_3D_out, proj_data_interpolator, index_converter);

      if (proj_data_out.set_segment(sino_3D_out) == Succeeded::no)
        return Succeeded::no;
    }
  return Succeeded::yes;
}

END_NAMESPACE_STIR
