//
//
/*
  Copyright (C) 2005- 2011, Hammersmith Imanet Ltd
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0
        
  See STIR/LICENSE.txt for details
*/
/*!
\file 
\ingroup projdata
\brief Implementation of functions to extension of direct sinograms in view direction

\author Kris Thielemans
\author Charalampos Tsoumpas
  
*/
#include "stir/Array.h"
#include "stir/SegmentBySinogram.h"
#include "stir/Sinogram.h"
#include "stir/ProjDataInfo.h"
#include "stir/IndexRange.h"
#include "stir/Bin.h"
#include "stir/round.h"

START_NAMESPACE_STIR

namespace detail 
{
  /* This function takes symmetries in the sinogram space into account
     to find data in the negative segment if necessary.
     However, it needs testing if it would work for non-direct sinograms.
  */
  inline static
  Array<2,float>
  extend_sinogram_in_views(const  Array<2,float>& sino_positive_segment,
                           const  Array<2,float>& sino_negative_segment,
                           const ProjDataInfo& proj_data_info,
                           const int min_view_extension, const int max_view_extension)
  {
    //* Check if projdata are from 0 to pi-phi
      bool min_is_extended=false;
    bool max_is_extended=false;
    BasicCoordinate<2,int> min_in, max_in;
    if (!sino_positive_segment.get_regular_range(min_in, max_in))
      {
        warning("input segment 0 should have a regular range"); 
      }

    const int org_min_view_num=min_in[1];
    const int org_max_view_num=max_in[1];

    const float min_phi = proj_data_info.get_phi(Bin(0,0,0,0));
    const float max_phi = proj_data_info.get_phi(Bin(0,max_in[1],0,0));

    const float sampling_phi = 
      proj_data_info.get_phi(Bin(0,1,0,0)) - min_phi;
    const int num_views_for_180 = round(_PI/sampling_phi);

    if (fabs(min_phi)< .01)
      {
        min_in[1]-=min_view_extension; 
        min_is_extended=true;                                   
      }
    if (fabs(max_phi-(_PI-sampling_phi))<.01) 
      {         
        max_in[1]+=max_view_extension;
        max_is_extended=true;           
      }


    IndexRange<2> extended_range(min_in, max_in);
    Array<2,float> input_extended_view(extended_range);   
                                
    if (!min_is_extended)
      warning("Minimum view of the original projdata is not 0");
    if (!max_is_extended)
      warning("Maximum view of the original projdata is not 180-sampling_phi");

    for (int view_num=min_in[1]; view_num<=max_in[1]; ++view_num)
      {
        bool use_extension=false;
        int symmetric_view_num=0;
        if (view_num<org_min_view_num && min_is_extended==true)
          {
            use_extension=true;
            symmetric_view_num= view_num + num_views_for_180;
          }
        else if (view_num>org_max_view_num && max_is_extended==true)
          {
            use_extension=true;
            symmetric_view_num = view_num - num_views_for_180;
          }

        if (!use_extension)
          input_extended_view[view_num]=
            sino_positive_segment[view_num]; 
        else
          {
            const int symmetric_min = std::max(min_in[2], -max_in[2]);
            const int symmetric_max = std::min(-min_in[2], max_in[2]);
            for (int tang_num=symmetric_min; tang_num<=symmetric_max; ++tang_num)
              input_extended_view[view_num][tang_num]=
                sino_negative_segment[symmetric_view_num][-tang_num];
            // now do extrapolation where we don't have data
            for (int tang_num=min_in[2]; tang_num<symmetric_min; ++tang_num)
              input_extended_view[view_num][tang_num] =
                input_extended_view[view_num][symmetric_min];
            for (int tang_num=symmetric_max+1; tang_num<=max_in[2]; ++tang_num)
              input_extended_view[view_num][tang_num] =
                input_extended_view[view_num][symmetric_max];
          }             
      } // loop over views
    return input_extended_view;
  }
} // end of namespace detail

Array<3,float>
extend_segment_in_views(const SegmentBySinogram<float>& sino, 
                        const int min_view_extension, const int max_view_extension)
{
  if (sino.get_segment_num()!=0)
    error("extend_segment with single segment works only for segment 0");

  BasicCoordinate<3,int> min, max;
                
  min[1]=sino.get_min_axial_pos_num();
  max[1]=sino.get_max_axial_pos_num();
  min[2]=sino.get_min_view_num();
  max[2]=sino.get_max_view_num();
  min[3]=sino.get_min_tangential_pos_num();
  max[3]=sino.get_max_tangential_pos_num();
  const IndexRange<3> out_range(min,max);
  Array<3,float> out(out_range);
  for (int ax_pos_num=min[1]; ax_pos_num <=max[1] ; ++ax_pos_num)
    {
      out[ax_pos_num] =
        detail::
        extend_sinogram_in_views(sino[ax_pos_num],sino[ax_pos_num], 
                                 *(sino.get_proj_data_info_sptr()),
                                 min_view_extension, max_view_extension);
    }
  return out;
}

Array<3,float>
extend_segment(const SegmentBySinogram<float>& segment, const int view_extension,
               const int axial_extension, const int tangential_extension)
{
  Array<3,float> out(segment);
  BasicCoordinate<3,int> min_dim, max_dim;
  min_dim[1] = segment.get_min_index() - axial_extension;
  min_dim[2] = segment[0].get_min_index() - view_extension;
  min_dim[3] = segment[0][0].get_min_index() - tangential_extension;
  max_dim[1] = segment.get_max_index() + axial_extension;
  max_dim[2] = segment[0].get_max_index() + view_extension;
  max_dim[3] = segment[0][0].get_max_index() + tangential_extension;
  out.grow(IndexRange<3>(min_dim, max_dim));

  // fill the axial extensions with the same entries from the border
  for (int axial_edge = 0; axial_edge < axial_extension; axial_edge++)
  {
    out[min_dim[1] + axial_edge] = out[min_dim[1] + axial_extension];
    out[max_dim[1] - axial_edge] = out[max_dim[1] - axial_extension];
  }

  // fill the view extensions by wrapping around
  for (int view_edge = 0; view_edge < view_extension; view_edge++)
  {
    for (int axial_pos = min_dim[1]; axial_pos <= max_dim[1]; axial_pos++)
    {
      // if views cover 360°, we can simply wrap around
      if (abs(segment.get_proj_data_info_sptr()->get_phi(Bin(0, segment.get_proj_data_info_sptr()->get_max_view_num(), 0, 0)) - 
              segment.get_proj_data_info_sptr()->get_phi(Bin(0, segment.get_proj_data_info_sptr()->get_min_view_num(), 0, 0))) > _PI)
      {
        out[axial_pos][min_dim[2] + view_edge] = out[axial_pos][max_dim[2] - 2 * view_extension + view_edge + 1];
        out[axial_pos][max_dim[2] - view_extension + 1 + view_edge] = out[axial_pos][min_dim[2] + view_extension + view_edge];
      }
      else
      { // if views cover 180°, the tangential positions need to be flipped
        const int sym_dim = std::min(abs(min_dim[3]), max_dim[3]);
        for (int tang_pos = -sym_dim; tang_pos <= sym_dim; tang_pos++)
        {
          out[axial_pos][min_dim[2] + view_edge][tang_pos] = out[axial_pos][max_dim[2] - 2 * view_extension + view_edge + 1][-tang_pos];
          out[axial_pos][max_dim[2] - view_extension + 1 + view_edge][tang_pos] = out[axial_pos][min_dim[2] + view_extension + view_edge][-tang_pos];
        }
        for (int tang_pos = min_dim[3]; tang_pos < -sym_dim; tang_pos++)
        { // fill in asymmetric tangential positions at the end by just picking the nearest existing element
          out[axial_pos][min_dim[2] + view_edge][tang_pos] = out[axial_pos][max_dim[2] - 2 * view_extension + view_edge + 1][sym_dim];
          out[axial_pos][max_dim[2] - view_extension + 1 + view_edge][tang_pos] = out[axial_pos][min_dim[2] + view_extension + view_edge][sym_dim];
        }
        for (int tang_pos = max_dim[3]; tang_pos > sym_dim; tang_pos--)
        { // fill in asymmetric tangential positions at the end by just picking the nearest existing element
          out[axial_pos][min_dim[2] + view_edge][tang_pos] = out[axial_pos][max_dim[2] - 2 * view_extension + view_edge + 1][-sym_dim];
          out[axial_pos][max_dim[2] - view_extension + 1 + view_edge][tang_pos] = out[axial_pos][min_dim[2] + view_extension + view_edge][-sym_dim];
        }
      }
    }
  }

  // fill tangential extension with same entries than boundary
  for (int tang_edge = 0; tang_edge < tangential_extension; tang_edge++)
  {
    for (int axial_pos = min_dim[1]; axial_pos <= max_dim[1]; axial_pos++)
    {
      for (int view = min_dim[2]; view <= max_dim[2]; view++)
      {
        out[axial_pos][view][min_dim[3] + tang_edge] = out[axial_pos][view][min_dim[3] + tangential_extension];
        out[axial_pos][view][max_dim[3] - tang_edge] = out[axial_pos][view][max_dim[3] - tangential_extension];
      }
    }
  }

  return out;
}

Array<2,float>
extend_sinogram_in_views(const Sinogram<float>& sino,
                         const int min_view_extension, const int max_view_extension)
{
  if (sino.get_segment_num()!=0)
    error("extend_segment with single segment works only for segment 0");

  return 
    detail::
    extend_sinogram_in_views(sino, sino,
                             *(sino.get_proj_data_info_sptr()),
                             min_view_extension, max_view_extension);
}

END_NAMESPACE_STIR
