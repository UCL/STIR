/*
    Copyright (C) 2026 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup recon_buildblock

  \brief Implementation of artificial scanner layout utilities for 3D sinograms.

*/

#include "stir/recon_buildblock/ArtificialScanner3D.h"
#include "stir/ProjDataInfo.h"
#include "stir/error.h"
#include "stir/warning.h"
#include <algorithm>
#include <cstdlib>

START_NAMESPACE_STIR

ArtificialScanner3DLayout
create_default_artificial_scanner3d_layout(const ProjDataInfo& proj_data_info)
{
  ArtificialScanner3DLayout layout;

  const int min_segment_num = proj_data_info.get_min_segment_num();
  const int max_segment_num = proj_data_info.get_max_segment_num();
  if (max_segment_num < min_segment_num)
    error("ArtificialScanner3D: invalid segment range (min=%d max=%d)", min_segment_num, max_segment_num);

  const int num_segments = max_segment_num - min_segment_num + 1;
  layout.segment_numbers.resize(num_segments);
  for (int i = 0; i < num_segments; ++i)
    layout.segment_numbers[i] = min_segment_num + i;

  layout.centre_index = num_segments / 2;
  if (min_segment_num <= 0 && 0 <= max_segment_num)
    layout.centre_index = -min_segment_num;
  if (layout.centre_index < 0 || layout.centre_index >= num_segments)
    error("ArtificialScanner3D: centre index out of range (%d for %d segments)", layout.centre_index, num_segments);
  layout.centre_segment_num = layout.segment_numbers[layout.centre_index];

  if (min_segment_num <= 0 && 0 <= max_segment_num)
    layout.reference_axial_count = proj_data_info.get_num_axial_poss(0);
  if (layout.reference_axial_count <= 0)
    {
      for (int i = 0; i < num_segments; ++i)
        layout.reference_axial_count
            = std::max(layout.reference_axial_count, proj_data_info.get_num_axial_poss(layout.segment_numbers[i]));
      warning("ArtificialScanner3D: segment 0 unavailable; using max axial count %d as reference",
              layout.reference_axial_count);
    }
  if (layout.reference_axial_count <= 0)
    error("ArtificialScanner3D: failed to derive a positive reference axial count");

  layout.measured_axial_counts.resize(num_segments);
  layout.target_axial_counts.resize(num_segments);
  layout.measured_offsets.resize(num_segments);

  for (int i = 0; i < num_segments; ++i)
    {
      const int segment_num = layout.segment_numbers[i];
      const int measured_count = proj_data_info.get_num_axial_poss(segment_num);
      if (measured_count <= 0)
        error("ArtificialScanner3D: non-positive axial count %d at segment %d", measured_count, segment_num);

      const int ring_diff_from_centre = std::abs(segment_num - layout.centre_segment_num);
      const int target_count = std::max(measured_count, layout.reference_axial_count + ring_diff_from_centre);

      layout.measured_axial_counts[i] = measured_count;
      layout.target_axial_counts[i] = target_count;
      layout.measured_offsets[i] = (target_count - measured_count) / 2;
    }

  return layout;
}

END_NAMESPACE_STIR
