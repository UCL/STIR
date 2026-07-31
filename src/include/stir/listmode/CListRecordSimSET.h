/*
    Copyright (C) 2019 University of Hull
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup listmode SimSET
  \brief Classes for listmode events for GATE simulated SimSET history file

  \author Efthimiou Nikos
*/

#ifndef __stir_listmode_CListRecordSimSET_H__
#define __stir_listmode_CListRecordSimSET_H__

#include "stir/listmode/CListEventScannerWithDiscreteDetectors.h"
#include "stir/Succeeded.h"

extern "C"
{
#include <LbTypes.h>
#include <Photon.h>
}

START_NAMESPACE_STIR

template <class ProjDataInfoT>
class CListEventSimSET : public CListEventScannerWithDiscreteDetectors<ProjDataInfoT>
{
public:
  inline CListEventSimSET(const shared_ptr<const ProjDataInfo>& proj_data_info_sptr)
      : CListEventScannerWithDiscreteDetectors<ProjDataInfoT>(proj_data_info_sptr)
  {}

  inline CListEventSimSET(shared_ptr<const ProjDataInfo> proj_data_info_sptr,
                          const DetectionPositionPair<>& det_pos_pair,
                          bool is_prompt_v)
      : CListEventScannerWithDiscreteDetectors<ProjDataInfoT>(proj_data_info_sptr),
        _det_pos_pair(det_pos_pair),
        _is_prompt(is_prompt_v)
  {}

  inline bool is_prompt() const override { return this->_is_prompt; }

  bool operator==(const CListEventSimSET& other) const
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

  inline void set_weight(float weight_arg) { weight = weight_arg; }

  virtual void get_detection_position(DetectionPositionPair<>& det_pos_pair_arg) const override
  {
    det_pos_pair_arg = this->_det_pos_pair;
  }

  virtual void set_detection_position(const DetectionPositionPair<>& det_pos_pair_arg) override
  {
    this->_det_pos_pair = det_pos_pair_arg;
  }

  virtual void set_lor(const LORAs2Points<float>& lor_arg, float tof_arg, float weight_arg) override
  {
    lor = lor_arg;
    tof = tof_arg;
    weight = weight_arg;
  }

  inline LORAs2Points<float> get_LOR() const override { return lor; }

  inline void get_bin(Bin& bin_arg, const ProjDataInfo& proj_data_info) const override
  {

    Bin bin = static_cast<ProjDataInfoT const&>(proj_data_info).get_bin(lor, tof);

    if (bin.get_bin_value() == 1)
      {
        bin.set_bin_value(weight);
      }
    bin_arg = bin;
  }

private:
  DetectionPositionPair<> _det_pos_pair;

  LORAs2Points<float> lor;

  float tof;

  float weight = 1;

  bool _is_prompt;
};

//! A class for storing and using a timing 'event' from a listmode file from the ECAT 8_32bit scanner
/*! \ingroup listmode
 */
class CListTimeSimSET : public ListTime
{
public:
  inline unsigned long get_time_in_millisecs() const { return static_cast<unsigned long>(time); }
  inline Succeeded set_time_in_millisecs(const unsigned long time_in_millisecs)
  {
    time = time_in_millisecs;
    return Succeeded::yes;
  }
  bool operator==(const CListTimeSimSET& other) const { return time == other.time; }
  inline bool is_time() const { return true; }
  uint32_t time;
};

//! A class for a general element of a listmode file for a Siemens scanner using the ROOT files
class CListRecordSimSET : public CListRecord
{
public:
  CListRecordSimSET(shared_ptr<const ProjDataInfo> proj_data_info_arg)
      : event_data(make_event_data(proj_data_info_arg)),
        proj_data_info_sptr(std::move(proj_data_info_arg))
  {
    radius = proj_data_info_sptr->get_scanner_sptr()->get_inner_ring_radius();
    half_ring_spacing = this->proj_data_info_sptr->get_scanner_sptr()->get_ring_spacing() / 2.F;
  }

  //! This record also has valid timing information.
  bool is_time() const override { return true; }
  //! This record represents a coincidence event.
  bool is_event() const override { return true; }

  //! Returns always true
  bool inline is_full_event() const;

  CListEvent& event() override { return *event_data; }
  const CListEvent& event() const override { return *event_data; }

  CListTimeSimSET& time() override { return time_data; }
  const CListTimeSimSET& time() const override { return time_data; }

  bool operator==(const CListRecordSimSET& e2) const { return event_data == e2.event_data && time_data == e2.time_data; }

  virtual Succeeded init_from_data(const PHG_DetectedPhoton& detectedPhotonBlue,
                                   const PHG_DetectedPhoton& detectedPhotonPink,
                                   const float weight,
                                   const bool is_prompt);

private:
  static std::unique_ptr<CListEventScannerWithDiscreteDetectorsBase>
  make_event_data(shared_ptr<const ProjDataInfo> proj_data_info);

  std::unique_ptr<CListEventScannerWithDiscreteDetectorsBase> event_data;

  CListTimeSimSET time_data;

  shared_ptr<const ProjDataInfo> proj_data_info_sptr;

  bool is_prompt_event = true;

  mutable float half_ring_spacing = 0.f;

  float radius;
};

END_NAMESPACE_STIR
#endif
