/*
    Copyright 2025, 2026 University Medical Center Groningen
    Copyright 2025 National Physical Laboratory

    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details.
*/

/*!

  \file
  \ingroup listmode
  \brief Declaration of class stir::CListEventPETSIRD and stir::CListRecordPETSIRD with supporting classes

  \author Nikos Efthimiou
  \author Daniel Deidda
*/

#ifndef __stir_listmode_CListRecordPETSIRD_H__
#define __stir_listmode_CListRecordPETSIRD_H__

#include "stir/listmode/CListEventScannerWithDiscreteDetectors.h"
#include "stir/listmode/CListRecord.h"
#include "stir/DetectionPositionPair.h"
#include "stir/PETSIRDInfo.h"
#include "stir/Succeeded.h"
#include "stir/format.h"

START_NAMESPACE_STIR

template <class ProjDataInfoT>
class CListEventPETSIRD : public CListEventScannerWithDiscreteDetectors<ProjDataInfoT>
{
public:
  //! Constructor which leaves \c det_pos_pair and \c is_prompts undefined.
  inline CListEventPETSIRD(shared_ptr<const ProjDataInfo> proj_data_info_sptr)
      : CListEventScannerWithDiscreteDetectors<ProjDataInfoT>(proj_data_info_sptr)
  {}

  inline CListEventPETSIRD(shared_ptr<const ProjDataInfo> proj_data_info_sptr,
                           const DetectionPositionPair<>& det_pos_pair,
                           bool is_prompt_v)
      : CListEventScannerWithDiscreteDetectors<ProjDataInfoT>(proj_data_info_sptr),
        _det_pos_pair(det_pos_pair),
        _is_prompt(is_prompt_v)
  {}

  inline bool is_prompt() const override { return this->_is_prompt; }

  bool operator==(const CListEventPETSIRD& other) const
  {
    if (this == &other)
      return true;

    return is_prompt() == other.is_prompt() && get_detection_position() == other.get_detection_position();
  }

  inline Succeeded set_prompt(const bool prompt) override
  {
    this->_is_prompt = prompt;
    return Succeeded::yes;
  }

  virtual void get_detection_position(DetectionPositionPair<>& det_pos_pair_arg) const override
  {
    det_pos_pair_arg = this->_det_pos_pair;
  }

  virtual void set_detection_position(const DetectionPositionPair<>& det_pos_pair_arg) override
  {
    this->_det_pos_pair = det_pos_pair_arg;
  }

private:
  DetectionPositionPair<> _det_pos_pair;
  bool _is_prompt;
};

class CListTimePETSIRD : public ListTime
{
public:
  inline unsigned long get_time_in_millisecs() const { return static_cast<unsigned long>(time); }
  inline Succeeded set_time_in_millisecs(const unsigned long time_in_millisecs)
  {
    time = time_in_millisecs;
    return Succeeded::yes;
  }
  bool operator==(const CListTimePETSIRD& other) const { return time == other.time; }
  inline bool is_time() const { return true; }
  uint32_t time;
};

/*!
  \brief Listmode record for PETSIRD coincidence events.

  PETSIRD stores coincidence events inside petsird::EventTimeBlock objects.
  The time associated with a PETSIRD event is therefore obtained from the
  enclosing EventTimeBlock, rather than from a separate time marker record.

  CListModeDataPETSIRD assigns this time to the record before returning it.
  Consequently, a CListRecordPETSIRD represents a coincidence event and also
  carries valid timing information.

  In this implementation, a record is both an event record and a time record,
  similar to CListRecordROOT.
*/
class CListRecordPETSIRD : public CListRecord
{
public:
  CListRecordPETSIRD(shared_ptr<const PETSIRDInfo> petsird_info_sptr, shared_ptr<const ProjDataInfo> proj_data_info_sptr)
      : event_data(make_event_data(proj_data_info_sptr)),
        petsird_info_sptr(std::move(petsird_info_sptr)),
        proj_data_info_sptr(std::move(proj_data_info_sptr))
  {}

  //! This record also has valid timing information.
  bool is_time() const override { return true; }
  //! This record represents a coincidence event.
  bool is_event() const override { return true; }

  CListEvent& event() override { return *event_data; }
  const CListEvent& event() const override { return *event_data; }

  CListTimePETSIRD& time() override { return time_data; }
  const CListTimePETSIRD& time() const override { return time_data; }

  bool operator==(const CListRecordPETSIRD& e2) const { return event_data == e2.event_data && time_data == e2.time_data; }

  Succeeded init_from_data(const petsird::CoincidenceEvent& event, const bool is_prompt)
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

private:
  static std::unique_ptr<CListEventScannerWithDiscreteDetectorsBase>
  make_event_data(shared_ptr<const ProjDataInfo> proj_data_info);

  std::unique_ptr<CListEventScannerWithDiscreteDetectorsBase> event_data;
  CListTimePETSIRD time_data;

  shared_ptr<const PETSIRDInfo> petsird_info_sptr;
  shared_ptr<const ProjDataInfo> proj_data_info_sptr;

  bool is_prompt_event = true;
};

END_NAMESPACE_STIR

#endif
