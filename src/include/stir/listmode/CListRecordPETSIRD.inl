/*      CListRecordPETSIRD.inl
        Coincidence Event Class for PETSIRD: Inline File

        Copyright 2025, 2026 UMCG
        Copyright 2025 National Physical Laboratory

    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details.

 */
/*!

  \file
  \ingroup listmode
  \brief Inline implementation of class stir::CListEventPETSIRD and stir::CListRecordPETSIRD with supporting classes

  \author Nikos Efthimiou
  \author Daniel Deidda
*/

#include "stir/LORCoordinates.h"
#include "stir/listmode/CListRecord.h"
#include "stir/ProjDataInfo.h"
#include "stir/Bin.h"
#include "stir/error.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoBlocksOnCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoGenericNoArcCorr.h"

START_NAMESPACE_STIR

void
CListEventPETSIRD::get_bin(Bin& bin, const ProjDataInfo& proj_data_info) const
{
  if (petsird_info_sptr->is_cylindrical_configuration_used() || petsird_info_sptr->is_block_configuration_used())
    {
      DetectionPositionPair<> det_pos_pair;
      get_detection_position_pair(det_pos_pair);
      //! Warning: assuming that STIR ProjDataInfo and PETSIRD TOF binning match. If you have tof_mashing this can be wrong.
      det_pos_pair.timing_pos() = static_cast<int>(m_tof_bin) + proj_data_info.get_min_tof_pos_num();
      // Try Blocks-on-Cylindrical first
      if (const auto* proj_blocks = dynamic_cast<const ProjDataInfoBlocksOnCylindricalNoArcCorr*>(&proj_data_info))
        {
          if (proj_blocks->get_bin_for_det_pos_pair(bin, det_pos_pair) == Succeeded::yes)
            bin.set_bin_value(1);
          else
            bin.set_bin_value(0);
        }
      // Else try Cylindrical
      else if (const auto* proj_cyl = dynamic_cast<const ProjDataInfoCylindricalNoArcCorr*>(&proj_data_info))
        {
          if (proj_cyl->get_bin_for_det_pos_pair(bin, det_pos_pair) == Succeeded::yes)
            bin.set_bin_value(1);
          else
            bin.set_bin_value(0);
        }
    }
  else
    {
      error("CListEventPETSIRD::Implement Generic Scanner.");
    }
}

LORAs2Points<float>
CListEventPETSIRD::get_LOR() const
{
}

END_NAMESPACE_STIR
