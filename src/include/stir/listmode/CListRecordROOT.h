/*
    Copyright (C) 2015-2016 University of Leeds
    Copyright (C) 2016 UCL
    Copyright (C) 2016-17, University of Hull
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup listmode
  \brief Classes for listmode events for GATE simulated ROOT data

  \author Efthimiou Nikos
  \author Harry Tsoumpas
*/

#ifndef __stir_listmode_CListRecordROOT_H__
#define __stir_listmode_CListRecordROOT_H__

#include "stir/listmode/CListRecord.h"
#include "stir/listmode/CListEventCylindricalScannerWithDiscreteDetectors.h"
#include "stir/Succeeded.h"
#include "stir/round.h"
#include "boost/static_assert.hpp"
#include "stir/DetectionPositionPair.h"
#include <boost/cstdint.hpp>
#include "stir/warning.h"

START_NAMESPACE_STIR

class CListEventROOT : public CListEventCylindricalScannerWithDiscreteDetectors
{
public:
  CListEventROOT(const shared_ptr<const ProjDataInfo>& proj_data_info);

  //! This routine returns the corresponding detector pair
  void get_detection_position(DetectionPositionPair<>&) const override;

  //! This routine sets in a coincidence event from detector "indices"
  void set_detection_position(const DetectionPositionPair<>&) override;

  //! \details This is the main function which transform GATE coordinates to STIR
  void init_from_data(const int& _ring1, const int& _ring2, const int& crystal1, const int& crystal2, const double& _delta_time);

  inline bool is_prompt() const override { return true; }

  double get_delta_time() const { return delta_time; }

private:
  //! First ring, in order to detector tangestial index
  int ring1;
  //! Second ring, in order to detector tangestial index
  int ring2;
  //! First detector, in order to detector tangestial index
  int det1;
  //! Second detector, in order to detector tangestial index
  int det2;
  //! The detection time difference, between the two photons.
  double delta_time;
#ifdef STIR_ROOT_ROTATION_AS_V4
  //! This is the number of detector we have to rotate in order to
  //! align GATE and STIR.
  int quarter_of_detectors;
#endif
};

//! A class for storing and using a timing 'event' from a listmode file from the ECAT 8_32bit scanner
/*! \ingroup listmode
 */
class CListTimeROOT : public ListTime
{
public:
  void init_from_data(double time1) { timeA = time1; }

  //! Returns always true
  bool is_time() const { return true; }
  //! Returns the detection time of the first photon
  //! in milliseconds.
  inline unsigned long get_time_in_millisecs() const override { return static_cast<unsigned long>(timeA * 1e3); }
  inline Succeeded set_time_in_millisecs(const unsigned long time_in_millisecs) override
  {
    warning("set_time_in_millisecs: Not implemented for ROOT files. Aborting.");
    return Succeeded::no;
  }

private:
  //!
  //! \brief timeA
  //! \details The detection time of the first of the two photons, in seconds
  double timeA;
};

//! A class for a general element of a listmode file for a Siemens scanner using the ROOT files
class CListRecordROOT : public CListRecord // currently no gating yet
{
public:
  //! Returns always true
  bool inline is_time() const override;
  //! Returns always true
  bool inline is_event() const override;

  CListEventROOT& event() override { return this->event_data; }

  const CListEventROOT& event() const override { return this->event_data; }

  CListTimeROOT& time() override { return this->time_data; }

  const CListTimeROOT& time() const override { return this->time_data; }

  bool operator==(const CListRecord& e2) const
  {
    return dynamic_cast<CListRecordROOT const*>(&e2) != 0 && raw[0] == dynamic_cast<CListRecordROOT const&>(e2).raw[0]
           && raw[1] == dynamic_cast<CListRecordROOT const&>(e2).raw[1];
  }

  CListRecordROOT(const shared_ptr<const ProjDataInfo>& proj_data_info_sptr)
      : event_data(proj_data_info_sptr)
  {}

  virtual Succeeded init_from_data(const int& ring1,
                                   const int& ring2,
                                   const int& crystal1,
                                   const int& crystal2,
                                   const double& time1,
                                   const double& delta_time,
                                   const int& event1,
                                   const int& event2)
  {
    /// \warning ROOT data are time and event at the same time.

    this->event_data.init_from_data(ring1, ring2, crystal1, crystal2, delta_time);

    this->time_data.init_from_data(time1);

    // We can make a singature raw based on the two events IDs.
    // It is pretty unique.
    raw[0] = event1;
    raw[1] = event2;

    return Succeeded::yes;
  }

private:
  CListEventROOT event_data;
  CListTimeROOT time_data;
  boost::int32_t raw[2]; // this raw field isn't strictly necessary, get rid of it?
};

END_NAMESPACE_STIR
#include "stir/listmode/CListRecordROOT.inl"
#endif
