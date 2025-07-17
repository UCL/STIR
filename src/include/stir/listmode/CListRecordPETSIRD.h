/* CListRecordPETSIRD.h

Coincidence Event Class for PETSIRD: Header File

     Copyright 2015 ETH Zurich, Institute of Particle Physics
     Copyright 2017 ETH Zurich, Institute of Particle Physics and Astrophysics
     Copyright 2020, 2022 Positrigo AG, Zurich
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

\author Jannis Fischer
\author Parisa Khateri
\author Markus Jehl
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
#include "types.h"

// #include "../../PETSIRD/cpp/generated/types.h"

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
  /*! Set the scanner */
  /*! Currently only used if the map is not set. */
  inline void set_scanner_sptr(shared_ptr<const Scanner> new_scanner_sptr) { scanner_sptr = new_scanner_sptr; }

  virtual bool is_valid_template(const ProjDataInfo&) const override { return true; }

  virtual bool is_prompt() const override { return _prompt; }

  virtual Succeeded set_prompt(const bool prompt) override
  {
    _prompt = prompt;
    return Succeeded::yes;
  }

  void set_PETSIRD_ranges(int _numberOfModules, int _numberOfElementsIndices)
  {
    numberOfModules = _numberOfModules;
    numberOfElementsIndices = _numberOfElementsIndices;
  }

  int numberOfModules;

  int numberOfElementsIndices;

  std::pair<int, int> det_0, det_1;

private:
  shared_ptr<const DetectorCoordinateMap> map_sptr;
  shared_ptr<const Scanner> scanner_sptr;
  bool _prompt;

  const DetectorCoordinateMap& map_to_use() const { return map_sptr ? *map_sptr : *this->scanner_sptr->get_detector_map_sptr(); }
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
  CListRecordPETSIRD() {}

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

  virtual Succeeded init_from_data(const petsird::CoincidenceEvent& data, bool is_prompt = true)
  {
    auto decodeElementAndModuleIndex
        = [](int linearIndex, int energyIndex, int numberOfElementsIndices, int numberOfModules) -> std::pair<int, int> {
      int reduced = (linearIndex - energyIndex) / numberOfModules;
      int moduleIndex = reduced / numberOfElementsIndices;
      int elementIndex = reduced % numberOfElementsIndices;
      return { elementIndex, moduleIndex };
    };

    event_data.det_0
        = decodeElementAndModuleIndex(data.detection_bins[0], 0, event_data.numberOfElementsIndices, event_data.numberOfModules);
    event_data.det_1
        = decodeElementAndModuleIndex(data.detection_bins[1], 0, event_data.numberOfElementsIndices, event_data.numberOfModules);

    std::cout << event_data.det_0.first << "  " << event_data.det_0.second << std::endl;
    std::cout << event_data.det_1.first << "  " << event_data.det_1.second << std::endl;
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
