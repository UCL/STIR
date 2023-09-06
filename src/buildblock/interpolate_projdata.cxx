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
#include "stir/IndexRange.h"
#include "stir/BasicCoordinate.h"
#include "stir/Sinogram.h"
#include "stir/SegmentBySinogram.h"
#include "stir/Succeeded.h"
#include "stir/numerics/BSplines.h"
#include "stir/numerics/BSplinesRegularGrid.h"
#include "stir/interpolate_projdata.h"
#include "stir/interpolate_axial_position.h"
#include "stir/extend_projdata.h"
#include "stir/numerics/sampling_functions.h"
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

    if (dynamic_cast<ProjDataInfoCylindricalNoArcCorr const *>(&proj_data_info) == NULL)
      error("make_non_interleaved_proj_data is only appropriate for non-arccorrected data");

    shared_ptr<ProjDataInfo> new_proj_data_info_sptr(
						     proj_data_info.clone());
    new_proj_data_info_sptr->
      set_num_views(proj_data_info.get_num_views()*2);
    return new_proj_data_info_sptr;
  }

  // access Sinogram element with wrap-around and boundary conditions
  static float sino_element(const Sinogram<float>& sinogram, const int view_num, const int tangential_pos_num)
  {
    assert(sinogram.get_min_view_num() == 0);
    const int num_views = sinogram.get_num_views();
    const int tang_pos_num = (view_num>=num_views? -1: 1)*tangential_pos_num;
    if (tang_pos_num < sinogram.get_min_tangential_pos_num() ||
	tang_pos_num > sinogram.get_max_tangential_pos_num())
      return 0.F;
    else
      return sinogram[view_num%num_views][tang_pos_num];
  }

  static void
  make_non_interleaved_sinogram(Sinogram<float>& out_sinogram,
                                const Sinogram<float>& in_sinogram)
  {
    if (is_null_ptr(dynamic_pointer_cast<const ProjDataInfoCylindricalNoArcCorr>(in_sinogram.get_proj_data_info_sptr())))
      error("make_non_interleaved_proj_data is only appropriate for non-arccorrected data");

    assert(out_sinogram.get_min_view_num() == 0);
    assert(in_sinogram.get_min_view_num() == 0);
    assert(out_sinogram.get_num_views() == in_sinogram.get_num_views()*2);
    assert(in_sinogram.get_segment_num() == 0);
    assert(out_sinogram.get_segment_num() == 0);

    for (int view_num = out_sinogram.get_min_view_num();
         view_num <= out_sinogram.get_max_view_num();
         ++view_num)
      {
        for (int tangential_pos_num = out_sinogram.get_min_tangential_pos_num()+1;
             tangential_pos_num <= out_sinogram.get_max_tangential_pos_num()-1;
             ++tangential_pos_num)
          {
            if ((view_num+tangential_pos_num)%2 == 0)
              {
                const int in_view_num =
                  view_num%2==0 ? view_num/2 : (view_num+1)/2;
                out_sinogram[view_num][tangential_pos_num] =
                  sino_element(in_sinogram, in_view_num, tangential_pos_num);
              }
            else
              {
                const int next_in_view = view_num/2+1;
                const int other_in_view = (view_num+1)/2;

                out_sinogram[view_num][tangential_pos_num] =
                  (sino_element(in_sinogram, view_num/2, tangential_pos_num) +
                   sino_element(in_sinogram, next_in_view, tangential_pos_num) +
                   sino_element(in_sinogram, other_in_view, tangential_pos_num-1) +
                   sino_element(in_sinogram, other_in_view, tangential_pos_num+1)
                   )/4;
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
  make_non_interleaved_segment(SegmentBySinogram<float>& out_segment,
                               const SegmentBySinogram<float>& in_segment)
  {
    if (is_null_ptr(dynamic_pointer_cast<const ProjDataInfoCylindricalNoArcCorr>(in_segment.get_proj_data_info_sptr())))
      error("make_non_interleaved_proj_data is only appropriate for non-arccorrected data");

    for (int axial_pos_num = out_segment.get_min_axial_pos_num();
         axial_pos_num <= out_segment.get_max_axial_pos_num();
         ++axial_pos_num)
      {
        Sinogram<float> out_sinogram = out_segment.get_sinogram(axial_pos_num);
        make_non_interleaved_sinogram(out_sinogram, 
                                      in_segment.get_sinogram(axial_pos_num));
        out_segment.set_sinogram(out_sinogram);
      }
  }

  static SegmentBySinogram<float>
  make_non_interleaved_segment(const ProjDataInfo& non_interleaved_proj_data_info,
                               const SegmentBySinogram<float>& in_segment)
  {
    SegmentBySinogram<float> out_segment =
      non_interleaved_proj_data_info.get_empty_segment_by_sinogram(in_segment.get_segment_num(),
    		  in_segment.get_timing_pos_num());

    make_non_interleaved_segment(out_segment, in_segment);
    return out_segment;
  }     

} // end namespace detail_interpolate_projdata
                                              

using namespace detail_interpolate_projdata;

Succeeded 
interpolate_projdata(ProjData& proj_data_out,
                     const ProjData& proj_data_in, const BSpline::BSplineType these_types,
                     const bool remove_interleaving)
{
  BasicCoordinate<3, BSpline::BSplineType> these_types_3; 
  these_types_3[1]=these_types_3[2]=these_types_3[3]=these_types;
  interpolate_projdata(proj_data_out,proj_data_in,these_types_3, remove_interleaving);
  return Succeeded::yes;
}

Succeeded 
interpolate_projdata(ProjData& proj_data_out,
                     const ProjData& proj_data_in,
                     const BasicCoordinate<3, BSpline::BSplineType> & these_types,
                     const bool remove_interleaving)
{
  const ProjDataInfo & proj_data_in_info =
    *proj_data_in.get_proj_data_info_sptr();
  const ProjDataInfo & proj_data_out_info =
    *proj_data_out.get_proj_data_info_sptr();

  if (typeid(proj_data_in_info) != typeid(proj_data_out_info))
    {
      error("interpolate_projdata needs both projection data  to be of the same type\n"
            "(e.g. both arc-corrected or both not arc-corrected)");
    }
  
  // check for the same ring radius
  // This is strictly speaking only necessary for non-arccorrected data, but
  // we leave it in for all cases.
  if (fabs(proj_data_in_info.get_scanner_ptr()->get_inner_ring_radius() -
           proj_data_out_info.get_scanner_ptr()->get_inner_ring_radius()) > 1)
    {
      error("interpolate_projdata needs both projection to be of a scanner with the same ring radius");
    }

  // initialise interpolator
  BSpline::BSplinesRegularGrid<3, float, float> proj_data_interpolator(these_types);
  for (int k=proj_data_out_info.get_min_tof_pos_num();
  k<=proj_data_out_info.get_max_tof_pos_num(); ++k)
  {
  SegmentBySinogram<float> segment = remove_interleaving ? 
    make_non_interleaved_segment(*(make_non_interleaved_proj_data_info(proj_data_in_info)), proj_data_in.get_segment_by_sinogram(0,k)) :
    proj_data_in.get_segment_by_sinogram(0,k);

  std::function<BasicCoordinate<3, double> (const BasicCoordinate<3, int>&)> index_converter;
  if (proj_data_in_info.get_scanner_sptr()->get_scanner_geometry()=="Cylindrical")
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

    const float in_sampling_phi
        = (proj_data_in_info.get_phi(Bin(0, 1, 0, 0)) - proj_data_in_info.get_phi(Bin(0, 0, 0, 0))) / (remove_interleaving ? 2 : 1);
    const float out_sampling_phi = proj_data_out_info.get_phi(Bin(0, 1, 0, 0)) - proj_data_out_info.get_phi(Bin(0, 0, 0, 0));
    offset[2] = (proj_data_out_info.get_phi(Bin(0, 0, 0, 0)) - proj_data_in_info.get_phi(Bin(0, 0, 0, 0))) / in_sampling_phi;
    step[2] = out_sampling_phi / in_sampling_phi;

    const float in_sampling_s = proj_data_in_info.get_sampling_in_s(Bin(0, 0, 0, 0));
    const float out_sampling_s = proj_data_out_info.get_sampling_in_s(Bin(0, 0, 0, 0));
    offset[3] = (proj_data_out_info.get_s(Bin(0, 0, 0, 0)) - proj_data_in_info.get_s(Bin(0, 0, 0, 0))) / in_sampling_s;
    step[3] = out_sampling_s / in_sampling_s;

    // define a function to translate indices in the output proj data to indices in input proj data
    index_converter = [&proj_data_out_info, offset, step](const BasicCoordinate<3, int>& index_out) 
        -> BasicCoordinate<3, double>
    {
      // translate to indices in input proj data
      BasicCoordinate<3, double> index_in;
      for (auto dim = 1; dim <= 3; dim++)
        index_in[dim] = index_out[dim] * step[dim] + offset[dim];

      return index_in;
    };
  }
  else
  { // for BlocksOnCylindrical, views and tangential positions are not subsampled and can be mapped 1:1
    if (proj_data_in_info.get_num_tangential_poss() != proj_data_out_info.get_num_tangential_poss())
    {
      error("Interpolation of BlocksOnCylindrical scanners assumes that number of tangential positions "
            "is the same in the downsampled scanner.");
    }
    if (proj_data_in_info.get_num_views() != proj_data_out_info.get_num_views())
    {
      error("Interpolation of BlocksOnCylindrical scanners assumes that number of views "
            "is the same in the downsampled scanner.");
    }

    // only extending in axial direction - an extension of 2 was found to be sufficient
    proj_data_interpolator.set_coef(extend_segment(segment, 0, 2, 0));

    auto m_offset = proj_data_in_info.get_m(Bin(0, 0, 0, 0));
    auto m_sampling = proj_data_in_info.get_sampling_in_m(Bin(0, 0, 0, 0));

    // confirm that proj_data_in has equidistant sampling in m
    for (auto axial_pos = proj_data_in_info.get_min_axial_pos_num(0); axial_pos <= proj_data_in_info.get_max_axial_pos_num(0); axial_pos++)
    {
      if (abs(m_sampling - proj_data_in_info.get_sampling_in_m(Bin(0, 0, axial_pos, 0))) > 1E-4)
        error("input projdata to interpolate_projdata are not equidistantly sampled in m.");
    }

    // define a function to translate indices in the output proj data to indices in input proj data
    index_converter = [&proj_data_out_info, m_offset, m_sampling](const BasicCoordinate<3, int>& index_out) 
        -> BasicCoordinate<3, double>
    {
      // translate index on output to coordinate
      auto bin = Bin(0 /* segment */, index_out[2] /* view */, index_out[1] /* axial pos */, index_out[3] /* tangential pos */);
      auto out_m = proj_data_out_info.get_m(bin);

      // translate to indices in input proj data
      BasicCoordinate<3, double> index_in;
      index_in[1] = (out_m - m_offset) / m_sampling;
      index_in[2] = index_out[2];
      index_in[3] = index_out[3];

      return index_in;
    };
  }
  
  SegmentBySinogram<float> sino_3D_out = proj_data_out.get_empty_segment_by_sinogram(0,false, k);
  sample_function_using_index_converter(sino_3D_out, proj_data_interpolator, index_converter);

  if (proj_data_out.set_segment(sino_3D_out) == Succeeded::no)
    return Succeeded::no;
  }
  return Succeeded::yes;
}

END_NAMESPACE_STIR
