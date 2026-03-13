/*
    Copyright (C) 2026 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup recon_buildblock

  \brief Utilities to derive a reusable artificial scanner axial layout for 3D sinograms.

*/

#ifndef __stir_recon_buildblock_ArtificialScanner3D_H__
#define __stir_recon_buildblock_ArtificialScanner3D_H__

#include "stir/common.h"
#include <vector>

START_NAMESPACE_STIR

class ProjDataInfo;

/*!
  \brief Compact per-segment layout used by artificial-scanner based 3D algorithms.

  The vectors are aligned by segment index \c iphi.
*/
struct ArtificialScanner3DLayout
{
  std::vector<int> segment_numbers;
  std::vector<int> measured_axial_counts;
  std::vector<int> target_axial_counts;
  std::vector<int> measured_offsets;
  int centre_index = 0;
  int centre_segment_num = 0;
  int reference_axial_count = 0;
};

/*!
  \brief Construct a default artificial scanner layout from projection-data geometry.

  The target axial count for each segment is:
  \code
    max(measured_axial_count, reference_axial_count + |seg - centre_segment|)
  \endcode
*/
ArtificialScanner3DLayout
create_default_artificial_scanner3d_layout(const ProjDataInfo& proj_data_info);

END_NAMESPACE_STIR

#endif
