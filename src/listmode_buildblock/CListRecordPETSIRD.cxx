/*
    Copyright 2026, University Medical Center Groningen
    Copyright 2026, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup listmode
  \brief Implementation of classes stir::CListRecordPETSIRD

  \author Nikos Efthimiou
  \author Kris Thielemans
*/

#include "stir/listmode/CListRecordPETSIRD.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoBlocksOnCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoGenericNoArcCorr.h"

START_NAMESPACE_STIR

std::unique_ptr<CListEvent>
CListRecordPETSIRD::make_event_data(shared_ptr<const ProjDataInfo> proj_data_info_sptr,
                                    DetectionPositionPair<>& det_pos_pair,
                                    bool& is_prompt_event)
{
  // construct event of type of current ProjDataInfo
  // Note: currently cumbersome due to change ProjDataInfo hierarchy.
  // The following is safe...
  // See https://github.com/UCL/STIR/commit/79bd05694091f7b08fb0237cb34bdbeedb256a45
  if ((proj_data_info_sptr->get_scanner_ptr()->get_scanner_geometry() == "Cylindrical")
      && (dynamic_cast<const ProjDataInfoCylindricalNoArcCorr*>(proj_data_info_sptr.get()) != nullptr))
    {
      return std::make_unique<CListEventPETSIRD<ProjDataInfoCylindricalNoArcCorr>>(
          proj_data_info_sptr, &det_pos_pair, &is_prompt_event);
    }

  if ((proj_data_info_sptr->get_scanner_ptr()->get_scanner_geometry() == "BlocksOnCylindrical")
      && (dynamic_cast<const ProjDataInfoBlocksOnCylindricalNoArcCorr*>(proj_data_info_sptr.get()) != nullptr))
    {
      return std::make_unique<CListEventPETSIRD<ProjDataInfoBlocksOnCylindricalNoArcCorr>>(
          proj_data_info_sptr, &det_pos_pair, &is_prompt_event);
    }

  if ((proj_data_info_sptr->get_scanner_ptr()->get_scanner_geometry() == "Generic")
      && (dynamic_cast<const ProjDataInfoGenericNoArcCorr*>(proj_data_info_sptr.get()) != nullptr))
    {
      return std::make_unique<CListEventPETSIRD<ProjDataInfoGenericNoArcCorr>>(
          proj_data_info_sptr, &det_pos_pair, &is_prompt_event);
    }

  error("Unsupported ProjDataInfo type in CListRecordPETSIRD::make_event_data");
  return nullptr;
}

END_NAMESPACE_STIR
