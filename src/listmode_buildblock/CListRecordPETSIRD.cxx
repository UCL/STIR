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
  \brief Implementation of class stir::CListRecordPETSIRD

  \author Nikos Efthimiou
  \author Kris Thielemans
*/

#include "stir/listmode/CListRecordPETSIRD.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoBlocksOnCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoGenericNoArcCorr.h"
#include "stir/format.h"

START_NAMESPACE_STIR

Succeeded
CListRecordPETSIRD::init_from_data(const petsird::CoincidenceEvent& event, const bool is_prompt)
{
  const auto scanner_info_sptr = petsird_info_sptr->get_petsird_scanner_info_sptr();

  const auto exp_det_0
      = petsird_helpers::expand_detection_bin(*scanner_info_sptr,
                                              0, // TODO type_of_module, currently we only support single module types.
                                              event.detection_bins[0]);
  const auto exp_det_1
      = petsird_helpers::expand_detection_bin(*scanner_info_sptr,
                                              0, // TODO type_of_module, currently we only support single module types.
                                              event.detection_bins[1]);
  auto it0 = petsird_info_sptr->get_petsird_to_stir_map()->find(exp_det_1);
  auto it1 = petsird_info_sptr->get_petsird_to_stir_map()->find(exp_det_0);
  if (it0 == petsird_info_sptr->get_petsird_to_stir_map()->end() || it1 == petsird_info_sptr->get_petsird_to_stir_map()->end())
    {
      error(format("get_stir_det_pos_from_PETSIRD_id: one or both PETSIRD ids (mod {}, elem {}, energy {}) not found",
                   exp_det_0.module_index,
                   exp_det_0.element_index,
                   exp_det_0.energy_index));
    }

  // Warning: this assumes that the PETSIRD TOF bins and the STIR ProjDataInfo
  // timing positions have the same binning/mashing and number of TOF bins.
  // If the STIR proj_data_info uses a different TOF mashing factor or TOF range,
  // this simple offset conversion is not valid.
  const DetectionPositionPair<> det_pos_pair(
      it0->second, it1->second, static_cast<int>(event.tof_idx) + this->proj_data_info_sptr->get_min_tof_pos_num());
  this->event_data->set_detection_position(det_pos_pair);
  this->event_data->set_prompt(is_prompt);
  return Succeeded::yes;
}

std::unique_ptr<CListEventScannerWithDiscreteDetectorsBase>
CListRecordPETSIRD::make_event_data(shared_ptr<const ProjDataInfo> proj_data_info_sptr)
{
  // construct event of type of current ProjDataInfo
  // Note: currently cumbersome due to change ProjDataInfo hierarchy.
  // The following is safe...
  // See https://github.com/UCL/STIR/commit/79bd05694091f7b08fb0237cb34bdbeedb256a45
  if ((proj_data_info_sptr->get_scanner_ptr()->get_scanner_geometry() == "Cylindrical")
      && (dynamic_cast<const ProjDataInfoCylindricalNoArcCorr*>(proj_data_info_sptr.get()) != nullptr))
    {
      return std::make_unique<CListEventPETSIRD<ProjDataInfoCylindricalNoArcCorr>>(proj_data_info_sptr);
    }

  if ((proj_data_info_sptr->get_scanner_ptr()->get_scanner_geometry() == "BlocksOnCylindrical")
      && (dynamic_cast<const ProjDataInfoBlocksOnCylindricalNoArcCorr*>(proj_data_info_sptr.get()) != nullptr))
    {
      return std::make_unique<CListEventPETSIRD<ProjDataInfoBlocksOnCylindricalNoArcCorr>>(proj_data_info_sptr);
    }

  if ((proj_data_info_sptr->get_scanner_ptr()->get_scanner_geometry() == "Generic")
      && (dynamic_cast<const ProjDataInfoGenericNoArcCorr*>(proj_data_info_sptr.get()) != nullptr))
    {
      return std::make_unique<CListEventPETSIRD<ProjDataInfoGenericNoArcCorr>>(proj_data_info_sptr);
    }

  error("Unsupported ProjDataInfo type in CListRecordPETSIRD::make_event_data");
  return nullptr;
}

END_NAMESPACE_STIR
