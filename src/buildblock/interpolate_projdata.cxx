//
//
/*
  Copyright (C) 2005 - 2009-10-27, Hammersmith Imanet Ltd
  Copyright (C) 2011-07-01 - 2011, Kris Thielemans
  This file is part of STIR.

  This file is free software; you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation; either version 2.1 of the License, or
  (at your option) any later version.
  
  This file is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Lesser General Public License for more details.
        
  See STIR/LICENSE.txt for details
*/
/*!
 \file
 \ingroup projdata
 \brief Perform B-Splines Interpolation of sinograms

 \author Charalampos Tsoumpas
 \author Kris Thielemans
  
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
#include "stir/extend_projdata.h"
#include "stir/numerics/sampling_functions.h"
#include "stir_experimental/motion/Transform3DObjectImageProcessor.h"
#include "stir_experimental/motion/transform_3d_object.h"
#include "stir_experimental/numerics/more_interpolators.h"
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

  static shared_ptr<ProjDataInfo>
  make_extended_proj_data_info(const ProjDataInfo& proj_data_info)
  {

    if (dynamic_cast<ProjDataInfoCylindricalNoArcCorr const *>(&proj_data_info) == NULL)
      error("make_extended_proj_data is only appropriate for non-arccorrected data");

    shared_ptr<ProjDataInfo> new_proj_data_info_sptr(
                             proj_data_info.clone());
    new_proj_data_info_sptr->
      set_num_views(proj_data_info.get_num_views()*2);
    return new_proj_data_info_sptr;
  }

  static shared_ptr<ProjDataInfo>
  transpose_make_non_interleaved_proj_data_info(const ProjDataInfo& proj_data_info)
  {

      // TODO: arc-correct?
      if (dynamic_cast<ProjDataInfoCylindricalNoArcCorr const *>(&proj_data_info) == NULL)
        error("make_non_interleaved_proj_data is only appropriate for non-arccorrected data");

    shared_ptr<ProjDataInfo> new_proj_data_info_sptr(
                             proj_data_info.clone());
    new_proj_data_info_sptr->
      set_num_views(proj_data_info.get_num_views()/2);
    return new_proj_data_info_sptr;
  }

  static void
  make_non_interleaved_sinogram(Sinogram<float>& out_sinogram,
                                const Sinogram<float>& in_sinogram)
  {
    if (dynamic_cast<ProjDataInfoCylindricalNoArcCorr const *>(in_sinogram.get_proj_data_info_ptr()) == NULL)
      error("make_non_interleaved_proj_data is only appropriate for non-arccorrected data");

    assert(out_sinogram.get_min_view_num() == 0);
    assert(in_sinogram.get_min_view_num() == 0);
    assert(out_sinogram.get_num_views() == in_sinogram.get_num_views()*2);
    assert(in_sinogram.get_segment_num() == 0);
    assert(out_sinogram.get_segment_num() == 0);

    const int in_num_views = in_sinogram.get_num_views();

    for (int view_num = out_sinogram.get_min_view_num();
         view_num <= out_sinogram.get_max_view_num();
         ++view_num)
      {
        // TODO don't put in outer tangential poss for now to avoid boundary stuff
        for (int tangential_pos_num = out_sinogram.get_min_tangential_pos_num()+1;
             tangential_pos_num <= out_sinogram.get_max_tangential_pos_num()-1;
             ++tangential_pos_num)
          {
            if ((view_num+tangential_pos_num)%2 == 0)
              {
                const int in_view_num =
                  view_num%2==0 ? view_num/2 : (view_num+1)/2;
                out_sinogram[view_num][tangential_pos_num] =
                  in_sinogram[in_view_num%in_num_views][(in_view_num>=in_num_views? -1: 1)*tangential_pos_num];
              }
            else
              {
                const int next_in_view = view_num/2+1;
                const int other_in_view = (view_num+1)/2;

                out_sinogram[view_num][tangential_pos_num] =
                  (in_sinogram[view_num/2][tangential_pos_num] +
                   in_sinogram[next_in_view%in_num_views][(next_in_view>=in_num_views ? -1 : 1)*tangential_pos_num] +
                   in_sinogram[other_in_view%in_num_views][(other_in_view>=in_num_views ? -1 : 1)*(tangential_pos_num-1)] +
                   in_sinogram[other_in_view%in_num_views][(other_in_view>=in_num_views ? -1 : 1)*(tangential_pos_num+1)]
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
  transpose_make_non_interleaved_sinogram(Sinogram<float>& out_sinogram,
                                const Sinogram<float>& in_sinogram)
  {
    if (dynamic_cast<ProjDataInfoCylindricalNoArcCorr const *>(in_sinogram.get_proj_data_info_ptr()) == NULL)
      error("make_non_interleaved_proj_data is only appropriate for non-arccorrected data");

    assert(out_sinogram.get_min_view_num() == 0);
    assert(in_sinogram.get_min_view_num() == 0);
    assert(out_sinogram.get_num_views() == in_sinogram.get_num_views()/2);
    assert(in_sinogram.get_segment_num() == 0);
    assert(out_sinogram.get_segment_num() == 0);

    const int in_num_views = in_sinogram.get_num_views();

    for (int view_num = out_sinogram.get_min_view_num();
         view_num <= out_sinogram.get_max_view_num();
         ++view_num)
      {
        // TODO don't put in outer tangential poss for now to avoid boundary stuff
        for (int tangential_pos_num = out_sinogram.get_min_tangential_pos_num()+1;
             tangential_pos_num <= out_sinogram.get_max_tangential_pos_num()-1;
             ++tangential_pos_num)
          {
            if ((view_num+tangential_pos_num)%2 == 0)
              {
                const int in_view_num =
                  view_num%2==0 ? view_num/2 : (view_num+1)/2;
                out_sinogram[view_num][tangential_pos_num] =
                  in_sinogram[in_view_num%in_num_views][(in_view_num>=in_num_views? -1: 1)*tangential_pos_num];
              }
            else
              {
                const int next_in_view = view_num/2+1;
                const int other_in_view = (view_num+1)/2;

                out_sinogram[view_num][tangential_pos_num] =
                  (in_sinogram[view_num/2][tangential_pos_num] +
                   in_sinogram[next_in_view%in_num_views][(next_in_view>=in_num_views ? -1 : 1)*tangential_pos_num] +
                   in_sinogram[other_in_view%in_num_views][(other_in_view>=in_num_views ? -1 : 1)*(tangential_pos_num-1)] +
                   in_sinogram[other_in_view%in_num_views][(other_in_view>=in_num_views ? -1 : 1)*(tangential_pos_num+1)]
                   )/4;
              }
          }
      }
  }


  static void
  make_non_interleaved_segment(SegmentBySinogram<float>& out_segment,
                               const SegmentBySinogram<float>& in_segment)
  {
    if (dynamic_cast<ProjDataInfoCylindricalNoArcCorr const *>(in_segment.get_proj_data_info_ptr()) == NULL)
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




  static void
  transpose_make_non_interleaved_segment(SegmentBySinogram<float>& out_segment,
                               const SegmentBySinogram<float>& in_segment)
  {
    if (dynamic_cast<ProjDataInfoCylindricalNoArcCorr const *>(in_segment.get_proj_data_info_ptr()) == NULL)
      error("make_non_interleaved_proj_data is only appropriate for non-arccorrected data");

    for (int axial_pos_num = out_segment.get_min_axial_pos_num();
         axial_pos_num <= out_segment.get_max_axial_pos_num();
         ++axial_pos_num)
      {
        Sinogram<float> out_sinogram = out_segment.get_sinogram(axial_pos_num);
        transpose_make_non_interleaved_sinogram(out_sinogram,
                                      in_segment.get_sinogram(axial_pos_num));
        out_segment.set_sinogram(out_sinogram);
      }
  }

  static SegmentBySinogram<float>
  make_non_interleaved_segment(const ProjDataInfo& non_interleaved_proj_data_info,
                               const SegmentBySinogram<float>& in_segment)
  {

    SegmentBySinogram<float> out_segment =
      non_interleaved_proj_data_info.get_empty_segment_by_sinogram(in_segment.get_segment_num());

    make_non_interleaved_segment(out_segment, in_segment);

    return out_segment;
  }     

  static SegmentBySinogram<float>
  transpose_make_non_interleaved_segment(const ProjDataInfo& compressed_proj_data_info,
                               const SegmentBySinogram<float>& in_segment)
  {

    SegmentBySinogram<float> out_segment =
      compressed_proj_data_info.get_empty_segment_by_sinogram(in_segment.get_segment_num());

    transpose_make_non_interleaved_segment(out_segment, in_segment);

    return out_segment;
  }
} // end namespace detail_interpolate_projdata
                                              

using namespace detail_interpolate_projdata;
  
Succeeded
interpolate_projdata(ProjData& proj_data_out,
                     const ProjData& proj_data_in, const BSpline::BSplineType these_types,
                     const bool remove_interleaving,
                     const bool use_view_offset)
{

  BasicCoordinate<3, BSpline::BSplineType> these_types_3; 

  these_types_3[1]=these_types_3[2]=these_types_3[3]=these_types;

  interpolate_projdata(proj_data_out,proj_data_in,these_types_3, remove_interleaving, use_view_offset);


  return Succeeded::yes;
}

Succeeded
interpolate_projdata(ProjData& proj_data_out,
                     const ProjData& proj_data_in,
                     const BasicCoordinate<3, BSpline::BSplineType> & these_types,
                     const bool remove_interleaving,
                     const bool use_view_offset)
{



  if (use_view_offset)
    warning("interpolate_projdata with use_view_offset is EXPERIMENTAL and NOT TESTED.");

  const ProjDataInfo & proj_data_in_info =
    *proj_data_in.get_proj_data_info_ptr();
  const ProjDataInfo & proj_data_out_info =
    *proj_data_out.get_proj_data_info_ptr();

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



  BSpline::BSplinesRegularGrid<3, float, float> proj_data_interpolator(these_types);

  BasicCoordinate<3, double>  offset,  step  ;

  // find relation between out_index and in_index such that they correspond to the same physical position
  // out_index * m_zoom + m_offset = in_index
  const float in_sampling_m = proj_data_in_info.get_sampling_in_m(Bin(0,0,0,0));
  const float out_sampling_m = proj_data_out_info.get_sampling_in_m(Bin(0,0,0,0));
  // offset in 'in' index units
  offset[1] = 
    (proj_data_in_info.get_m(Bin(0,0,0,0)) -
     proj_data_out_info.get_m(Bin(0,0,0,0))) / in_sampling_m;
  step[1]=
    out_sampling_m/in_sampling_m;
                
  const float in_sampling_phi = 
    (proj_data_in_info.get_phi(Bin(0,1,0,0)) - proj_data_in_info.get_phi(Bin(0,0,0,0))) /
    (remove_interleaving ? 2 : 1);

  const float out_sampling_phi = 
    proj_data_out_info.get_phi(Bin(0,1,0,0)) - proj_data_out_info.get_phi(Bin(0,0,0,0));
 
  const float out_view_offset = 
    use_view_offset
    ? proj_data_out_info.get_scanner_ptr()->get_default_intrinsic_tilt()
    : 0.F;
  const float in_view_offset = 
    use_view_offset
    ? proj_data_in_info.get_scanner_ptr()->get_default_intrinsic_tilt()
    : 0.F;
  offset[2] = 
    (proj_data_in_info.get_phi(Bin(0,0,0,0)) + in_view_offset - proj_data_out_info.get_phi(Bin(0,0,0,0)) - out_view_offset) / in_sampling_phi;
  step[2] =
    out_sampling_phi/in_sampling_phi;
        
  const float in_sampling_s = proj_data_in_info.get_sampling_in_s(Bin(0,0,0,0));
  const float out_sampling_s = proj_data_out_info.get_sampling_in_s(Bin(0,0,0,0));
  offset[3] = 
    (proj_data_out_info.get_s(Bin(0,0,0,0)) -
     proj_data_in_info.get_s(Bin(0,0,0,0))) / in_sampling_s;
  step[3]=
    out_sampling_s/in_sampling_s;
        
  // initialise interpolator
  if (remove_interleaving)

  {


    shared_ptr<ProjDataInfo> non_interleaved_proj_data_info_sptr =
      make_non_interleaved_proj_data_info(proj_data_in_info);

    const SegmentBySinogram<float> non_interleaved_segment =
      make_non_interleaved_segment(*non_interleaved_proj_data_info_sptr,
                                           proj_data_in.get_segment_by_sinogram(0));
    //    display(non_interleaved_segment, non_interleaved_segment.find_max(),"non-inter");

    Array<3,float> extended = 
      extend_segment_in_views(non_interleaved_segment, 2, 2);
    for (int z=extended.get_min_index(); z<= extended.get_max_index(); ++z)
      {
        for (int y=extended[z].get_min_index(); y<= extended[z].get_max_index(); ++y)
          {
            const int old_min = extended[z][y].get_min_index();
            const int old_max = extended[z][y].get_max_index();
            extended[z][y].grow(old_min-1, old_max+1);
            extended[z][y][old_min-1] = extended[z][y][old_min];
            extended[z][y][old_max+1] = extended[z][y][old_max];
          }
      }
    std::cerr << "MAX UP "<< extended.find_max() << '\n';
    std::cerr << "MIN UP "<< extended.find_min() << '\n';
    proj_data_interpolator.set_coef(extended);
  }
  else
  {
    Array<3,float> extended = 
      extend_segment_in_views(proj_data_in.get_segment_by_sinogram(0), 2, 2);
    for (int z=extended.get_min_index(); z<= extended.get_max_index(); ++z)
      {
        for (int y=extended[z].get_min_index(); y<= extended[z].get_max_index(); ++y)
          {
            const int old_min = extended[z][y].get_min_index();
            const int old_max = extended[z][y].get_max_index();
            extended[z][y].grow(old_min-1, old_max+1);
            extended[z][y][old_min-1] = extended[z][y][old_min];
            extended[z][y][old_max+1] = extended[z][y][old_max];
          }
      }
     std::cerr << "MAX UP "<< extended.find_max() << '\n';
     std::cerr << "MIN UP "<< extended.find_min() << '\n';
    proj_data_interpolator.set_coef(extended);
  }
        
  // now do interpolation

  SegmentBySinogram<float> sino_3D_out = proj_data_out.get_empty_segment_by_sinogram(0) ;
  sample_function_on_regular_grid(sino_3D_out, proj_data_interpolator, offset, step);

  proj_data_out.set_segment(sino_3D_out);

  if (proj_data_out.set_segment(sino_3D_out) == Succeeded::no)
    return Succeeded::no;          
  return Succeeded::yes;
}




Succeeded
interpolate_projdata_pull(ProjData& proj_data_out,
                     const ProjData& proj_data_in,
                     const bool remove_interleaving,
                     const bool use_view_offset)
{


    SegmentBySinogram<float> sino_3D_out = proj_data_out.get_empty_segment_by_sinogram(0) ;
  if (use_view_offset)
    warning("interpolate_projdata with use_view_offset is EXPERIMENTAL and NOT TESTED.");

  const ProjDataInfo & proj_data_in_info =
    *proj_data_in.get_proj_data_info_ptr();
  const ProjDataInfo & proj_data_out_info =
    *proj_data_out.get_proj_data_info_ptr();

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





  BasicCoordinate<3, double>  offset,  step  ;

  // find relation between out_index and in_index such that they correspond to the same physical position
  // out_index * m_zoom + m_offset = in_index
  const float in_sampling_m = proj_data_in_info.get_sampling_in_m(Bin(0,0,0,0));
  const float out_sampling_m = proj_data_out_info.get_sampling_in_m(Bin(0,0,0,0));
  // offset in 'in' index units
  offset[1] =
    (proj_data_in_info.get_m(Bin(0,0,0,0)) -
     proj_data_out_info.get_m(Bin(0,0,0,0))) / in_sampling_m;
  step[1]=
    out_sampling_m/in_sampling_m;

  const float in_sampling_phi =
    (proj_data_in_info.get_phi(Bin(0,1,0,0)) - proj_data_in_info.get_phi(Bin(0,0,0,0))) /
    (remove_interleaving ? 2 : 1);

  const float out_sampling_phi =
    proj_data_out_info.get_phi(Bin(0,1,0,0)) - proj_data_out_info.get_phi(Bin(0,0,0,0));

  const float out_view_offset =
    use_view_offset
    ? proj_data_out_info.get_scanner_ptr()->get_default_intrinsic_tilt()
    : 0.F;
  const float in_view_offset =
    use_view_offset
    ? proj_data_in_info.get_scanner_ptr()->get_default_intrinsic_tilt()
    : 0.F;
  offset[2] =
    (proj_data_in_info.get_phi(Bin(0,0,0,0)) + in_view_offset - proj_data_out_info.get_phi(Bin(0,0,0,0)) - out_view_offset) / in_sampling_phi;
  step[2] =
    out_sampling_phi/in_sampling_phi;

  const float in_sampling_s = proj_data_in_info.get_sampling_in_s(Bin(0,0,0,0));
  const float out_sampling_s = proj_data_out_info.get_sampling_in_s(Bin(0,0,0,0));
  offset[3] =
    (proj_data_out_info.get_s(Bin(0,0,0,0)) -
     proj_data_in_info.get_s(Bin(0,0,0,0))) / in_sampling_s;
  step[3]=
    out_sampling_s/in_sampling_s;

  std::cerr << "PULL - OFFSET"<< offset[1] << "," << offset[2] << "," << offset[3] << '\n';
  std::cerr << "PULL - STEP"<< step[1] << "," << step[2] << "," << step[3] << '\n';
  // initialise interpolator
  if (remove_interleaving)

  {


    shared_ptr<ProjDataInfo> non_interleaved_proj_data_info_sptr =
      make_non_interleaved_proj_data_info(proj_data_in_info);

    const SegmentBySinogram<float> non_interleaved_segment =
      make_non_interleaved_segment(*non_interleaved_proj_data_info_sptr,
                                           proj_data_in.get_segment_by_sinogram(0));
    //    display(non_interleaved_segment, non_interleaved_segment.find_max(),"non-inter");

    Array<3,float> extended =
      extend_segment_in_views(non_interleaved_segment, 2, 2);
    for (int z=extended.get_min_index(); z<= extended.get_max_index(); ++z)
      {
        for (int y=extended[z].get_min_index(); y<= extended[z].get_max_index(); ++y)
          {
            const int old_min = extended[z][y].get_min_index();
            const int old_max = extended[z][y].get_max_index();
            extended[z][y].grow(old_min-1, old_max+1);
            extended[z][y][old_min-1] = extended[z][y][old_min];
            extended[z][y][old_max+1] = extended[z][y][old_max];
          }
      }

    sample_function_on_regular_grid_pull(sino_3D_out,extended, offset, step);
    proj_data_out.set_segment(sino_3D_out);
    if (proj_data_out.set_segment(sino_3D_out) == Succeeded::no)
      return Succeeded::no;
  }
  else
  {
    Array<3,float> extended =
      extend_segment_in_views(proj_data_in.get_segment_by_sinogram(0), 2, 2);
    for (int z=extended.get_min_index(); z<= extended.get_max_index(); ++z)
      {
        for (int y=extended[z].get_min_index(); y<= extended[z].get_max_index(); ++y)
          {
            const int old_min = extended[z][y].get_min_index();
            const int old_max = extended[z][y].get_max_index();
            extended[z][y].grow(old_min-1, old_max+1);
            extended[z][y][old_min-1] = extended[z][y][old_min];
            extended[z][y][old_max+1] = extended[z][y][old_max];
          }
      }


    sample_function_on_regular_grid_pull(sino_3D_out,extended, offset, step);
    proj_data_out.set_segment(sino_3D_out);
    if (proj_data_out.set_segment(sino_3D_out) == Succeeded::no)
      return Succeeded::no;

  }

  return Succeeded::yes;
}




Succeeded
interpolate_projdata_push(ProjData& proj_data_out,
                     const ProjData& proj_data_in,
                     const bool remove_interleaving,
                     const bool use_view_offset)
{


    SegmentBySinogram<float> sino_3D_out = proj_data_out.get_empty_segment_by_sinogram(0) ;

  if (use_view_offset)
    warning("interpolate_projdata with use_view_offset is EXPERIMENTAL and NOT TESTED.");

  const ProjDataInfo & proj_data_in_info =
    *proj_data_in.get_proj_data_info_ptr();
  const ProjDataInfo & proj_data_out_info =
    *proj_data_out.get_proj_data_info_ptr();

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



  BasicCoordinate<3, double>  offset,  step  ;

  // find relation between out_index and in_index such that they correspond to the same physical position
  // out_index * m_zoom + m_offset = in_index
  const float in_sampling_m = proj_data_in_info.get_sampling_in_m(Bin(0,0,0,0));
  const float out_sampling_m = proj_data_out_info.get_sampling_in_m(Bin(0,0,0,0));
  // offset in 'in' index units
  offset[1] =
    (proj_data_out_info.get_m(Bin(0,0,0,0)) -
     proj_data_in_info.get_m(Bin(0,0,0,0))) / out_sampling_m; //divide by sampling: conversion from mm to voxel units
  step[1]=
    in_sampling_m/out_sampling_m;

  const float out_sampling_phi =
    (proj_data_out_info.get_phi(Bin(0,1,0,0)) - proj_data_out_info.get_phi(Bin(0,0,0,0))) /
    (remove_interleaving ? 2 : 1);

  const float in_sampling_phi =
    proj_data_in_info.get_phi(Bin(0,1,0,0)) - proj_data_in_info.get_phi(Bin(0,0,0,0));

  const float out_view_offset =
    use_view_offset
    ? proj_data_out_info.get_scanner_ptr()->get_default_intrinsic_tilt()
    : 0.F;
  const float in_view_offset =
    use_view_offset
    ? proj_data_in_info.get_scanner_ptr()->get_default_intrinsic_tilt()
    : 0.F;
  offset[2] =
    (proj_data_out_info.get_phi(Bin(0,0,0,0)) + out_view_offset - proj_data_in_info.get_phi(Bin(0,0,0,0)) - in_view_offset) / out_sampling_phi;
  step[2] =
    in_sampling_phi/out_sampling_phi;

  const float out_sampling_s = proj_data_out_info.get_sampling_in_s(Bin(0,0,0,0));
  const float in_sampling_s = proj_data_in_info.get_sampling_in_s(Bin(0,0,0,0));
  offset[3] =
    (proj_data_in_info.get_s(Bin(0,0,0,0)) -
     proj_data_out_info.get_s(Bin(0,0,0,0))) / out_sampling_s;
  step[3]=
    in_sampling_s/out_sampling_s;

  std::cerr << "PUSH - OFFSET:"<< offset[1] << "," << offset[2] << "," << offset[3] << '\n';
  std::cerr << "PUSH - STEP:"<< step[1] << "," << step[2] << "," << step[3] << '\n';

  // initialise interpolator
  if (remove_interleaving)

  {

    //we need to create an 'extended output'
    shared_ptr<ProjDataInfo> non_interleaved_proj_data_info_sptr =
      make_non_interleaved_proj_data_info(proj_data_out_info);

    const SegmentBySinogram<float> non_interleaved_segment =
      make_non_interleaved_segment(*non_interleaved_proj_data_info_sptr,
                                           proj_data_out.get_segment_by_sinogram(0));
    //    display(non_interleaved_segment, non_interleaved_segment.find_max(),"non-inter");

    Array<3,float> extended =
      extend_segment_in_views(non_interleaved_segment, 2, 2);
    for (int z=extended.get_min_index(); z<= extended.get_max_index(); ++z)
      {
        for (int y=extended[z].get_min_index(); y<= extended[z].get_max_index(); ++y)
          {
            const int old_min = extended[z][y].get_min_index();
            const int old_max = extended[z][y].get_max_index();
            extended[z][y].grow(old_min-1, old_max+1);
            extended[z][y][old_min-1] = extended[z][y][old_min];
            extended[z][y][old_max+1] = extended[z][y][old_max];

          }
      }
    for(int segm_num = proj_data_in.get_min_segment_num(); segm_num <= proj_data_in.get_max_segment_num(); ++segm_num)
    {
     SegmentBySinogram<float> sino_3D_in = proj_data_in.get_segment_by_sinogram(segm_num);
      sample_function_on_regular_grid_push(extended,sino_3D_in, offset, step);
      //now we need to do first the transpose of extended -> size non_interleaved_proj_data
      // then we do the transpose of remove interleaving -> size projdata_out
    }
    proj_data_out.set_segment(sino_3D_out);
    if (proj_data_out.set_segment(sino_3D_out) == Succeeded::no)
        return Succeeded::no;

  }
  else
  {
    Array<3,float> extended = extend_segment_in_views(proj_data_out.get_segment_by_sinogram(0), 2, 2);

    int z_dim = extended[0][0].size_all();
    int y_dim = extended[0].size_all()/z_dim;
    int x_dim = extended.size_all()/(z_dim*y_dim);
    std::cout<<"ext:" << x_dim << "x" <<  y_dim << "x" << z_dim << '\n';
  /* for (int z=extended.get_min_index(); z<= extended.get_max_index(); ++z)
      {
        for (int y=extended[z].get_min_index(); y<= extended[z].get_max_index(); ++y)
          {
            const int old_min = extended[z][y].get_min_index();
            const int old_max = extended[z][y].get_max_index();
            extended[z][y].grow(old_min-1, old_max+1);
            extended[z][y][old_min-1] = extended[z][y][old_min];
            extended[z][y][old_max+1] = extended[z][y][old_max];
          }
      }*/

    z_dim = extended[0][0].size_all();
    y_dim = extended[0].size_all()/z_dim;
    x_dim = extended.size_all()/(z_dim*y_dim);
   std::cout<<"ext2222:" << x_dim << "x" <<  y_dim << "x" << z_dim << '\n';

    for(int segm_num = proj_data_in.get_min_segment_num(); segm_num <= proj_data_in.get_max_segment_num(); ++segm_num)
    {
     SegmentBySinogram<float> sino_3D_in = proj_data_in.get_segment_by_sinogram(segm_num);

     //here the output of the push is 'extended'
      sample_function_on_regular_grid_push(extended,sino_3D_in, offset, step);
      //create extended projdata info
      shared_ptr<ProjDataInfo> extended_proj_data_info_sptr(proj_data_out_info.clone());
     // std::cout<< "views" << extended_proj_data_info_sptr->get_num_views() << '\n';

     //create SegmentBySinogram with extended
      SegmentBySinogram<float> extended_segment_sino(extended, extended_proj_data_info_sptr, 0);

      std::cout<<"ext before compress:" << extended_segment_sino.get_num_views() << "x" <<  extended_segment_sino.get_num_tangential_poss() << '\n';


     // set the number of views correctly
      extended_proj_data_info_sptr->set_num_views(extended_segment_sino.get_num_views());

      // i'm passing extended_segment_sino that is equivalent to the array 'extended' and i need to compress it
     Array<3,float> out = transpose_extend_segment_in_views(extended_segment_sino,2, 2); // here we do the tranpose : extended -> sino_out

     /*for (int z=out.get_min_index(); z<= out.get_max_index(); ++z)
        {
          for (int y=out[z].get_min_index(); y<= out[z].get_max_index(); ++y)
            {
              const int old_min = out[z][y].get_min_index();
              const int old_max = out[z][y].get_max_index();
              out[z][y].grow(old_min+1, old_max-1);
              out[z][y][old_min-1] = out[z][y][old_min];
              out[z][y][old_max+1] = out[z][y][old_max];
            }
        }*/

      std::cout<<"correct output:" << sino_3D_out.get_num_views() << "x" <<  sino_3D_out.get_num_tangential_poss() << '\n';

      SegmentBySinogram<float> compressed_output(out, extended_proj_data_info_sptr, 0);
       std::cout<<"compressed output:" <<  compressed_output.get_num_views() << "x" <<   compressed_output.get_num_tangential_poss() << '\n';
      proj_data_out.set_segment(compressed_output);
      if (proj_data_out.set_segment(compressed_output) == Succeeded::no)
        return Succeeded::no;

    }
  }

   return Succeeded::yes;
}


END_NAMESPACE_STIR
