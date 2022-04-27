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
 \brief Perform B-Splines Interpolation of axial position

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
  
  float m_offset_in=proj_data_in_info.get_m(Bin(0,0,0,0));
  for (int segment=proj_data_out.get_min_segment_num();segment<=proj_data_out.get_max_segment_num();segment++)
      for (int axial_pos=proj_data_out.get_min_axial_pos_num(segment); axial_pos<=proj_data_out.get_max_axial_pos_num(segment);axial_pos++)
      {
          Bin bin(segment,0,axial_pos,0);
          int relative_pos=0;
          float m = proj_data_out_info.get_m(bin);
          float in_m_sampling= proj_data_in_info.get_sampling_in_m(bin);
          relative_pos=abs(round((m -m_offset_in)/ in_m_sampling));
           Sinogram<float> sino= proj_data_out.get_empty_sinogram(axial_pos,segment);
          for (int view=proj_data_out.get_min_view_num(); view<=proj_data_out.get_max_view_num();view++)
              for (int tan=proj_data_out.get_min_tangential_pos_num(); tan<=proj_data_out.get_max_tangential_pos_num();tan++)
              {
                  sino[view][tan]=proj_data_in.get_sinogram(relative_pos,0)[view][tan]; //proj_data_interpolator(relative_pos));
              }
          if (proj_data_out.set_sinogram(sino) == Succeeded::no)
              return Succeeded::no;
      }
  return Succeeded::yes;
}

END_NAMESPACE_STIR
