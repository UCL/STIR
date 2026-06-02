/*

Coincidence Event Class for PETSIRD: Header File

     Copyright 2025, University Medical Center Groningen
     Copyright 2025 National Physical Laboratory

     Licensed under the Apache License, Version 2.0 (the "License");
     you may not use this file except in compliance with the License.
     You may obtain a copy of the License at

             http://www.apache.org/licenses/LICENSE-2.0

     Unless required by applicable law or agreed to in writing, software
     distributed under the License is distributed on an "AS IS" BASIS,
     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     See the License for the specific language governing permissions and
     limitations under the License.

 */

/*!

\file CListRecordPETSIRD
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
#include "stir/Succeeded.h"
#include "stir/ByteOrderDefine.h"

#include "stir/DetectorCoordinateMap.h"
#include "stir/PETSIRDInfo.h"

START_NAMESPACE_STIR

template <class ProjDataInfoT>
class CListEventPETSIRD : public CListEventScannerWithDiscreteDetectors<ProjDataInfoT>
{
public:
  inline CListEventPETSIRD(shared_ptr<const ProjDataInfo> proj_data_info_sptr,
                          DetectionPositionPair<>* det_pos_pair,
                          bool* is_prompt)
      : CListEventScannerWithDiscreteDetectors<ProjDataInfoT>(proj_data_info_sptr),
        det_pos_pair_ptr(det_pos_pair),
        is_prompt_ptr(is_prompt)
  {}

  // inline void get_bin(Bin& bin, const ProjDataInfo& proj_data_info) const override;

  inline bool is_prompt() const override { return *this->is_prompt_ptr; }

  bool operator==(const CListEventPETSIRD& other) const
  {
    if (this == &other)
      return true;

    return is_prompt() == other.is_prompt() && get_detection_position() == other.get_detection_position();
  }

  inline Succeeded set_prompt(const bool prompt) override
  {
    *this->is_prompt_ptr = prompt;
    return Succeeded::yes;
  }

  virtual void get_detection_position(DetectionPositionPair<>& det_pos_pair) const override
  {
    det_pos_pair = *this->det_pos_pair_ptr;
  }

  virtual void set_detection_position(const DetectionPositionPair<>& det_pos_pair) override
  {
    *this->det_pos_pair_ptr = det_pos_pair;
  }

private:
  DetectionPositionPair<>* det_pos_pair_ptr = nullptr;
  bool* is_prompt_ptr = nullptr;
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

class CListRecordPETSIRD : public CListRecord
{
public:
  CListRecordPETSIRD(shared_ptr<const PETSIRDInfo> petsird_info_sptr, shared_ptr<const ProjDataInfo> proj_data_info_sptr)
      : event_data(make_event_data(proj_data_info_sptr, this->det_pos_pair, this->is_prompt_event)),
        petsird_info_sptr(std::move(petsird_info_sptr)), 
        proj_data_info_sptr(std::move(proj_data_info_sptr))
  {}

  bool is_time() const override { return true; /*time_data.is_time();*/ }

  bool is_event() const override { return true; }

  CListEvent& event() override { return *event_data; }
  const CListEvent& event() const override { return *event_data; }

  CListTimePETSIRD& time() override { return time_data; }
  const CListTimePETSIRD& time() const override { return time_data; }

  bool operator==(const CListRecordPETSIRD& e2) const { return event_data == e2.event_data && time_data == e2.time_data; }

  Succeeded init_from_data(petsird::CoincidenceEvent& event, const bool is_prompt = true)
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
        error("get_stir_det_pos_from_PETSIRD_id: one or both PETSIRD ids not found",
              exp_det_0.module_index,
              exp_det_0.element_index,
              exp_det_0.energy_index);
      }

    // Warning: this assumes that the PETSIRD TOF bins and the STIR ProjDataInfo
    // timing positions have the same binning/mashing and number of TOF bins.
    // If the STIR proj_data_info uses a different TOF mashing factor or TOF range,
    // this simple offset conversion is not valid.
    this->det_pos_pair = DetectionPositionPair<>(
        it0->second, it1->second, static_cast<int>(event.tof_idx) + this->proj_data_info_sptr->get_min_tof_pos_num());

    is_prompt_event = is_prompt;
    return Succeeded::yes;
  }

private:
  static std::unique_ptr<CListEvent> make_event_data(shared_ptr<const ProjDataInfo> proj_data_info,
                                                    DetectionPositionPair<>& det_pos_pair,
                                                    bool& is_prompt_event);

  std::unique_ptr<CListEvent> event_data;
  CListTimePETSIRD time_data;

  shared_ptr<const PETSIRDInfo> petsird_info_sptr;
  shared_ptr<const ProjDataInfo> proj_data_info_sptr;

  bool is_prompt_event = true;
  DetectionPositionPair<> det_pos_pair;
};

END_NAMESPACE_STIR

// #include "CListRecordPETSIRD.inl"

#endif
