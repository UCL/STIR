/* CListRecordPETSIRD.h

Coincidence Event Class for PETSIRD: Header File

     Copyright 2025, UMCG
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
#include "stir/Succeeded.h"
#include "stir/ByteOrderDefine.h"

#include "stir/DetectorCoordinateMap.h"
#include "stir/PETSIRDInfo.h"

START_NAMESPACE_STIR

/*!
Provides interface of the record class to STIR by implementing get_LOR(). It uses an optional map from detector indices to
coordinates to specify LORAs2Points from given detection pair indices.

\ingroup listmode
*/

class CListEventPETSIRD : public CListEvent
{
public:
  inline CListEventPETSIRD(shared_ptr<const PETSIRDInfo> petsird_info_sptr)
      : petsird_info_sptr(std::move(petsird_info_sptr))
  {}

  //! Returns LOR corresponding to the given event.
  inline LORAs2Points<float> get_LOR() const override;

  inline void get_bin(Bin& bin, const ProjDataInfo& proj_data_info) const override;

  inline bool is_valid_template(const ProjDataInfo&) const override { return true; }

  inline bool is_prompt() const override { return m_prompt; }

  inline Succeeded set_prompt(const bool prompt) override
  {
    m_prompt = prompt;
    return Succeeded::yes;
  }

  inline void set_from_petsird(const petsird::CoincidenceEvent& event)
  {
    set_expanded_detection_bins(
        petsird_helpers::expand_detection_bin(*petsird_info_sptr->get_petsird_scanner_info_sptr(),
                                              0, // TODO type_of_module, currently we only support single module types.
                                              event.detection_bins[0]),
        petsird_helpers::expand_detection_bin(*petsird_info_sptr->get_petsird_scanner_info_sptr(),
                                              0, // TODO type_of_module, currently we only support single module types.
                                              event.detection_bins[1]),
        event.tof_idx); // + (this->proj_data_info_sptr->get_min_tof_pos_num()));
  }

  inline void set_expanded_detection_bins(const petsird_helpers::ExpandedDetectionBin& det0,
                                          const petsird_helpers::ExpandedDetectionBin& det1,
                                          const uint32_t tof_idx)
  {
    exp_det_0 = det0;
    exp_det_1 = det1;
    m_tof_bin = tof_idx;
  }

  inline void set_tof_bin(const int32_t value) { m_tof_bin = value; }

  inline void get_detection_position_pair(DetectionPositionPair<>& det_pos) const
  {
    // const-friendly lookup
    auto it0 = petsird_info_sptr->get_petsird_to_stir_map()->find(exp_det_1);
    auto it1 = petsird_info_sptr->get_petsird_to_stir_map()->find(exp_det_0);
    if (it0 == petsird_info_sptr->get_petsird_to_stir_map()->end() || it1 == petsird_info_sptr->get_petsird_to_stir_map()->end())
      {
        error("get_stir_det_pos_from_PETSIRD_id: one or both PETSIRD ids not found",
              exp_det_0.module_index,
              exp_det_0.element_index,
              exp_det_0.energy_index);
      }

    det_pos.pos1() = it0->second; // copy of DetectionPosition<>
    det_pos.pos2() = it1->second; // copy of DetectionPosition<>

    // std::cout << det_pos.pos1().tangential_coord() << ", " << det_pos.pos1().axial_coord() << ", "
    //           << det_pos.pos1().radial_coord() << std::endl;
    // std::cout << det_pos.pos2().tangential_coord() << ", " << det_pos.pos2().axial_coord() << ", "
    //           << det_pos.pos2().radial_coord() << std::endl;

    // auto s_it0 = petsird_info_sptr->get_stir_to_petsird_map()->find(it0->second);
    // auto s_it1 = petsird_info_sptr->get_stir_to_petsird_map()->find(it1->second);

    // std::cout << "DET 0: module " << s_it0->second.module_index << " element " << s_it0->second.element_index << " energy " <<
    //  s_it0->second.energy_index << std::endl; std::cout << "PETSIRD DET 0: module " << exp_det_0.module_index << " element " <<
    //  exp_det_0.element_index << " energy " << exp_det_0.energy_index << std::endl; std::cout << "DET 1: module " <<
    //  s_it1->second.module_index << " element " << s_it1->second.element_index << " energy " << s_it1->second.energy_index <<
    //  std::endl; std::cout << "PETSIRD DET 1: module " << exp_det_1.module_index << " element " << exp_det_1.element_index << "
    //  energy " << exp_det_1.energy_index << std::endl;

    det_pos.timing_pos() = static_cast<int>(m_tof_bin); //+
  }

private:
  bool m_prompt;
  petsird_helpers::ExpandedDetectionBin exp_det_0, exp_det_1;
  uint32_t m_tof_bin;
  shared_ptr<const PETSIRDInfo> petsird_info_sptr;
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
  inline bool is_time() const { return true; }
  uint32_t time;
};

class CListRecordPETSIRD : public CListRecord
{
public:
  CListRecordPETSIRD(shared_ptr<const PETSIRDInfo> petsird_info_sptr)
      : event_data(petsird_info_sptr)
  {}

  bool is_time() const override { return true; /*time_data.is_time();*/ }

  bool is_event() const override { return true; }

  CListEventPETSIRD& event() override { return event_data; }
  const CListEventPETSIRD& event() const override { return event_data; }

  CListTimePETSIRD& time() override { return time_data; }
  const CListTimePETSIRD& time() const override { return time_data; }

  bool operator==(const CListRecordPETSIRD& e2) const
  {
    // return dynamic_cast<CListRecordPETSIRD const*>(&e2) != 0 && raw == static_cast<CListRecordPETSIRD const&>(e2).r;
  }

  virtual Succeeded init_from_data(petsird::CoincidenceEvent& event, const bool is_prompt = true)
  {
    // event_data.set_expanded_detection_bins(det0, det1, tof_idx);
    event_data.set_from_petsird(event);
    event_data.set_prompt(is_prompt);
    return Succeeded::yes;
  }

private:
  CListEventPETSIRD event_data;
  CListTimePETSIRD time_data;
};

END_NAMESPACE_STIR

#include "CListRecordPETSIRD.inl"

#endif
