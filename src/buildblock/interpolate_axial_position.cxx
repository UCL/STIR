//
//
/*
    Copyright (C) 2022 National Physical Laboratory
    Copyright (C) 2022 University College London
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0
        
  See STIR/LICENSE.txt for details
*/
/*!
 \file
 \ingroup projdata
 \brief Perform B-Splines Interpolation of axial position. At present, it uses nearest neighbour interpolation in segment 0 if projdata_in only

 \author Daniel Deidda
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
#include "stir/interpolate_axial_position.h"
#include "stir/extend_projdata.h"
#include "stir/numerics/sampling_functions.h"
#include <typeinfo>
#ifdef STIR_OPENMP
#include <omp.h>
#endif
#include "stir/num_threads.h"

START_NAMESPACE_STIR

Succeeded 
interpolate_axial_position(ProjData& proj_data_out,
                     const ProjData& proj_data_in)
{
  const ProjDataInfo & proj_data_in_info =
    *proj_data_in.get_proj_data_info_sptr();
  const ProjDataInfo & proj_data_out_info =
    *proj_data_out.get_proj_data_info_sptr();

  if (typeid(proj_data_in_info) != typeid(proj_data_out_info))
    {
      error("interpolate_axial_position needs both projection data  to be of the same type\n"
            "(e.g. both arc-corrected or both not arc-corrected)");
    }
  if (fabs(proj_data_in_info.get_scanner_ptr()->get_inner_ring_radius() -
           proj_data_out_info.get_scanner_ptr()->get_inner_ring_radius()) > 1)
    {
      error("interpolate_axial_position needs both projection to be of a scanner with the same ring radius");
    }
//  create maps for the m coordinate depending on segment and axial_position
  VectorWithOffset<VectorWithOffset<float> > m_in(proj_data_in.get_min_segment_num(),proj_data_in.get_max_segment_num());
  VectorWithOffset<VectorWithOffset<float> > m_out(proj_data_out.get_min_segment_num(),proj_data_out.get_max_segment_num());

  for (int segment=proj_data_in.get_min_segment_num();segment<=proj_data_in.get_max_segment_num();segment++)
  {
      m_in.at(segment).resize(proj_data_in.get_min_axial_pos_num(segment),proj_data_in.get_max_axial_pos_num(segment));

      for (int axial_pos=proj_data_in.get_min_axial_pos_num(segment); axial_pos<=proj_data_in.get_max_axial_pos_num(segment);axial_pos++)
      {
          Bin bin(segment,0,axial_pos,0);
          m_in.at(segment).at(axial_pos)=proj_data_in_info.get_m(bin);
      }
  }

  for (int segment=proj_data_out.get_min_segment_num();segment<=proj_data_out.get_max_segment_num();segment++)
  {
      m_out.at(segment).resize(proj_data_out.get_min_axial_pos_num(segment),proj_data_out.get_max_axial_pos_num(segment));

      for (int axial_pos=proj_data_out.get_min_axial_pos_num(segment); axial_pos<=proj_data_out.get_max_axial_pos_num(segment);axial_pos++)
      {
          Bin bin(segment,0,axial_pos,0);
          m_out.at(segment).at(axial_pos)=proj_data_out_info.get_m(bin);
      }
  }

//now calculate the difference between the upsampled m (m_out) and the direct m (m_in) to be used for the nearest-neigbour interpolation
  for (int segment=proj_data_out.get_min_segment_num();segment<=proj_data_out.get_max_segment_num();segment++)
      for (int axial_pos=proj_data_out.get_min_axial_pos_num(segment); axial_pos<=proj_data_out.get_max_axial_pos_num(segment);axial_pos++)
      {
          VectorWithOffset<float> diff(proj_data_in.get_min_axial_pos_num(0),proj_data_in.get_max_axial_pos_num(0));
          int axial_pos_in=0;

          if (proj_data_in_info==proj_data_out_info)
              axial_pos_in=axial_pos;
          else{
              for (auto it=m_in.at(0).begin();it!=m_in.at(0).end();it++)
              {
                 diff.at(it-m_in.at(0).begin())=abs(m_out.at(segment).at(axial_pos) - *it);
              }
              auto result=std::min_element(diff.begin(),diff.end());
              axial_pos_in=std::distance(diff.begin(), result);
          }
          
           Sinogram<float> sino= proj_data_out.get_empty_sinogram(axial_pos,segment);
           const auto sino_in=proj_data_in.get_sinogram(axial_pos_in,0);
#ifdef STIR_OPENMP
#  if _OPENMP <201107
                      #pragma omp parallel for
#  else
                      #pragma omp parallel for collapse(2) schedule(dynamic)
#  endif
#endif
          for (int view=proj_data_out.get_min_view_num(); view<=proj_data_out.get_max_view_num();view++)
              for (int tan=proj_data_out.get_min_tangential_pos_num(); tan<=proj_data_out.get_max_tangential_pos_num();tan++)
              {
                  sino[view][tan]=sino_in[view][tan];
              }
          if (proj_data_out.set_sinogram(sino) == Succeeded::no)
              return Succeeded::no;
      }
  return Succeeded::yes;
}

END_NAMESPACE_STIR
