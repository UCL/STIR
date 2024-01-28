//
//
/*
  Copyright (C) 2005- 2011, Hammersmith Imanet Ltd
  Copyright 2023, Positrigo AG, Zurich
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
\author Markus Jehl
*/
#include "stir/Array.h"
#include "stir/SegmentBySinogram.h"
#include "stir/Sinogram.h"
#include "stir/ProjDataInfo.h"
#include "stir/IndexRange.h"
#include "stir/Bin.h"
#include "stir/round.h"
#include "stir/warning.h"
#include "stir/error.h"

START_NAMESPACE_STIR

/* This function takes symmetries into account to extend segments in any direction.
   However, it needs testing if it works for non-direct sinograms.
*/
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

  // check, whether the projection data cover 180° or 360°
  bool flip_views = false;
  bool extend_without_wrapping = false;
  float min_phi=_PI, max_phi=-_PI;
  for (auto view = segment.get_proj_data_info_sptr()->get_min_view_num(); view <= segment.get_proj_data_info_sptr()->get_max_view_num(); view++)
  {
    auto phi = segment.get_proj_data_info_sptr()->get_phi(Bin(0, view, 0, 0));
    if (phi < min_phi)
      min_phi = phi;
    if (phi > max_phi)
      max_phi = phi;
  }
  const auto phi_range = max_phi - min_phi;
  const auto average_phi_sampling = phi_range / (segment.get_proj_data_info_sptr()->get_num_views() - 1);
  // check if 360 or 180 degrees
  // use a rather large tolerance to cope with non-uniform sampling in BlocksOnCylindrical
  if (abs(phi_range - 2 * _PI) < 5 * average_phi_sampling)
    flip_views = false; // if views cover 360°, we can simply wrap around
  else if ((abs(phi_range - _PI) < 5 * average_phi_sampling) && (segment.get_segment_num() == 0))
    flip_views = true;  // if views cover 180°, the tangential positions need to be flipped
  else
  {
    extend_without_wrapping = true;
    warning("Extending ProjData by wrapping only works for view coverage of 180° or 360°. Instead, just extending with nearest neighbour.");
  }

  // fill the view extensions by wrapping around
  for (int view_edge = 0; view_edge < view_extension; view_edge++)
  {
    for (int axial_pos = min_dim[1]; axial_pos <= max_dim[1]; axial_pos++)
    {
      if (extend_without_wrapping)
      {
        out[axial_pos][min_dim[2] + view_edge] = out[axial_pos][min_dim[2] + view_extension];
        out[axial_pos][max_dim[2] - view_extension] = out[axial_pos][max_dim[2] - view_extension];
      }
      else if (flip_views)
      {
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
      else
      {
        out[axial_pos][min_dim[2] + view_edge] = out[axial_pos][max_dim[2] - 2 * view_extension + view_edge + 1];
        out[axial_pos][max_dim[2] - view_extension + 1 + view_edge] = out[axial_pos][min_dim[2] + view_extension + view_edge];
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

END_NAMESPACE_STIR
