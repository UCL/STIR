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

#include "stir/listmode/CListRecord.h"
#include "stir/DetectionPositionPair.h"
#include "stir/Succeeded.h"
#include "stir/ByteOrderDefine.h"
#include "boost/cstdint.hpp"

#include "stir/DetectorCoordinateMap.h"

START_NAMESPACE_STIR

/*!
Provides interface of the record class to STIR by implementing get_LOR(). It uses an optional map from detector indices to
coordinates to specify LORAs2Points from given detection pair indices.

\ingroup listmode
*/

class CListEventPETSIRD : public CListEvent
{
public:
  inline CListEventPETSIRD() {}

  //! Returns LOR corresponding to the given event.
  inline LORAs2Points<float> get_LOR() const override;

  //! Override the default implementation
  inline void get_bin(Bin& bin, const ProjDataInfo& proj_data_info) const override;

  inline void set_map_sptr(shared_ptr<const DetectorCoordinateMap> new_map_sptr) { map_sptr = new_map_sptr; }

  inline void set_petsird_to_stir_map(shared_ptr<PETSIRDToSTIRDetectorIndexMap> new_map) { petsird_to_stir = new_map; }

  inline bool is_valid_template(const ProjDataInfo&) const override { return true; }

  inline bool is_prompt() const override { return m_prompt; }

  inline Succeeded set_prompt(const bool prompt) override
  {
    m_prompt = prompt;
    return Succeeded::yes;
  }

  inline void set_expanded_detection_bins(const petsird_helpers::ExpandedDetectionBin& det0,
                                       const petsird_helpers::ExpandedDetectionBin& det1, 
                                      const uint32_t tof_idx) 
  {
    exp_det_0 = det0;
    exp_det_1 = det1;
    m_tof_bin = tof_idx; 
  }

  inline void set_tof_bin(const uint32_t value) { m_tof_bin = value; }

  inline stir::DetectionPosition<>
  get_stir_det_pos_from_PETSIRD_id(const petsird_helpers::ExpandedDetectionBin& exp_det_bin) const;

private:
  shared_ptr<const DetectorCoordinateMap> map_sptr = nullptr;
  shared_ptr<PETSIRDToSTIRDetectorIndexMap> petsird_to_stir = nullptr;

  bool m_prompt;
  petsird_helpers::ExpandedDetectionBin exp_det_0, exp_det_1;
  uint32_t m_tof_bin;
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
  CListRecordPETSIRD()
  {
  }

  // ~CListRecordPETSIRD() override {}

  bool is_time() const override { return true; /*time_data.is_time();*/ }

  bool is_event() const override { return true; }

  CListEventPETSIRD& event() override { return event_data; }
  const CListEventPETSIRD& event() const override { /*return event_data;*/ }

  CListTimePETSIRD& time() override { return time_data; }
  const CListTimePETSIRD& time() const override { return time_data; }

  bool operator==(const CListRecordPETSIRD& e2) const
  {
    // return dynamic_cast<CListRecordPETSIRD const*>(&e2) != 0 && raw == static_cast<CListRecordPETSIRD const&>(e2).r;
  }

  virtual Succeeded init_from_data(const petsird_helpers::ExpandedDetectionBin& det0,
                                   const petsird_helpers::ExpandedDetectionBin& det1,
                                   const uint32_t tof_idx,
                                   const bool is_prompt = true)
  {
    event_data.set_expanded_detection_bins(det0, det1, tof_idx);
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
