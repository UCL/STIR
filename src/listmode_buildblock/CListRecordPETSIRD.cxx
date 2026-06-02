/*
    Copyright 2026, University Medical Center Groningen
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup listmode
  \brief Implementation of classes CListRecordPETSIRD

  \author Nikos Efthimiou
*/

#include "stir/listmode/CListRecordPETSIRD.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoBlocksOnCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoGenericNoArcCorr.h"

START_NAMESPACE_STIR

std::unique_ptr<CListEvent>
CListRecordPETSIRD::make_event_data(shared_ptr<const ProjDataInfo> proj_data_info_sptr,
                                    const DetectionPositionPair<>& det_pos_pair,
                                    const bool& is_prompt_event)
{
  if (dynamic_cast<const ProjDataInfoCylindricalNoArcCorr*>(proj_data_info_sptr.get()) != nullptr)
    {
      return std::make_unique<CListEventPETSIRD<ProjDataInfoCylindricalNoArcCorr>>(
          proj_data_info_sptr, &det_pos_pair, &is_prompt_event);
    }

  if (dynamic_cast<const ProjDataInfoBlocksOnCylindricalNoArcCorr*>(proj_data_info_sptr.get()) != nullptr)
    {
      return std::make_unique<CListEventPETSIRD<ProjDataInfoBlocksOnCylindricalNoArcCorr>>(
          proj_data_info_sptr, &det_pos_pair, &is_prompt_event);
    }

  if (dynamic_cast<const ProjDataInfoGenericNoArcCorr*>(proj_data_info_sptr.get()) != nullptr)
    {
      return std::make_unique<CListEventPETSIRD<ProjDataInfoGenericNoArcCorr>>(
          proj_data_info_sptr, &det_pos_pair, &is_prompt_event);
    }

  error("Unsupported ProjDataInfo type in CListRecordPETSIRD::make_event_data");
  return nullptr;
}

END_NAMESPACE_STIR