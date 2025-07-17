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
#include "boost/make_shared.hpp"

#include "../../PETSIRD/cpp/generated/types.h"

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

  //! Returns 0 if event is prompt and 1 if delayed
  inline bool is_prompt() const override { return true; }

  inline void set_map_sptr(shared_ptr<const DetectorCoordinateMap> new_map_sptr) { map_sptr = new_map_sptr; }
  /*! Set the scanner */
  /*! Currently only used if the map is not set. */
  inline void set_scanner_sptr(shared_ptr<const Scanner> new_scanner_sptr) { scanner_sptr = new_scanner_sptr; }

  virtual bool is_valid_template(const ProjDataInfo&) const override { return true; }

private:
  shared_ptr<const DetectorCoordinateMap> map_sptr;
  shared_ptr<const Scanner> scanner_sptr;

  const DetectorCoordinateMap& map_to_use() const { return map_sptr ? *map_sptr : *this->scanner_sptr->get_detector_map_sptr(); }
};

//! Class for record with time data using PETSIRD bitfield definition
/*! \ingroup listmode */
class CListTimePETSIRD : public ListTime
{
public:
  inline unsigned long get_time_in_millisecs() const { /*return static_cast<unsigned long>(time);*/ }
  inline Succeeded set_time_in_millisecs(const unsigned long time_in_millisecs)
  {
    // time = ((boost::uint64_t(1) << 49) - 1) & static_cast<boost::uint64_t>(time_in_millisecs);
    return Succeeded::yes;
  }
  inline bool is_time() const { /*return type; */ }
};

class CListRecordPETSIRD : public CListRecord
{
public:
  CListRecordPETSIRD() {}

  ~CListRecordPETSIRD() override {}

  bool is_time() const override { /*return time_data.is_time();*/ }

  bool is_event() const override { /*return !time_data.is_time();*/ }

  ListEvent& event() override { return event_data; }
  const ListEvent& event() const override { return event_data; }

  ListTime& time() override { return time_data; }
  const ListTime& time() const override { return time_data; }

  CListEventPETSIRD& event_PETSIRD() { return event_data; }

  const CListEventPETSIRD& event_PETSIRD() const { return event_data; }

  // virtual bool operator==(const CListRecordPETSIRD& e2) const
  // {
  //   // return dynamic_cast<CListRecordPETSIRD const*>(&e2) != 0 && raw == static_cast<CListRecordPETSIRD const&>(e2).r;
  // }

  // inline bool is_prompt() const override { /*return event_data.is_prompt();*/ }

  Succeeded init_from_data_ptr(const petsird::CoincidenceEvent&)
  {
    // assert(size_of_record >= 8);
    // std::copy(data_ptr, data_ptr + 8, reinterpret_cast<char*>(&raw)); // TODO necessary for operator==
    // if (do_byte_swap)
    //   ByteOrder::swap_order(raw);
    return Succeeded::yes;
  }

private:
  CListEventPETSIRD event_data;
  CListTimePETSIRD time_data;
};

END_NAMESPACE_STIR

#include "CListRecordPETSIRD.inl"

#endif
