/*
    Copyright (C) 2015-2016 University of Leeds
    Copyright (C) 2016 UCL
    Copyright (C) 2016, University of Hull
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

START_NAMESPACE_STIR

class CListEventROOT : public CListEventCylindricalScannerWithDiscreteDetectors
{
public:

    CListEventROOT(const shared_ptr<Scanner>& scanner_sptr);

    //! This routine returns the corresponding detector pair
    virtual void get_detection_position(DetectionPositionPair<>&) const;

    //! This routine sets in a coincidence event from detector "indices"
    virtual void set_detection_position(const DetectionPositionPair<>&);

    //! \details This is the main function which transform GATE coordinates to STIR
    void init_from_data(const int &_ring1, const int &_ring2,
                             const int &crystal1, const int &crystal2);

    inline bool is_prompt() const
    { return true; }

    bool inline is_swapped() const
    { return swapped; }

private:
    //! First ring, in order to detector tangestial index
    int ring1;
    //! Second ring, in order to detector tangestial index
    int ring2;
    //! First detector, in order to detector tangestial index
    int det1;
    //! Second detector, in order to detector tangestial index
    int det2;
    //! Indicates if swap segments
    bool swapped;
    //! This is the number of detector we have to rotate in order to
    //! align GATE and STIR.
    int quarter_of_detectors;
};

//! A class for storing and using a timing 'event' from a listmode file from the ECAT 8_32bit scanner
/*! \ingroup listmode
 */
class CListTimeROOT : public CListTime
{
public:
    void init_from_data(float _timeA, float _delta_time)
    {
        timeA = _timeA;
        delta_time = _delta_time;
    }

    //! Returns always true
    bool is_time() const
    { return true; }
    //! Returns the detection time of the first photon
    //! in milliseconds.
    inline unsigned long  get_time_in_millisecs() const
    { return timeA * 1e3; }
    //! Get the detection time of the first photon
    //! in milliseconds
    inline unsigned long get_timeA_in_millisecs() const
    { return timeA * 1e3; }
    //! Get the detection time of the second photon
    //! in milliseconds
    inline unsigned long get_timeB_in_millisecs() const
    { return (delta_time - timeA) * 1e3; }
    //! Get the delta Time between the two events
    inline unsigned long get_delta_time_in_millisecs() const
    { return delta_time * 1e3; }
    //! Get delta time in picoseconds
    inline unsigned long get_delta_time_in_picosecs() const
    { return (timeB - timeA) * 1e12; }

    inline Succeeded set_time_in_millisecs(const unsigned long time_in_millisecs)
    {
        warning("set_time_in_millisecs: Not implemented for ROOT files. Aborting.");
        return Succeeded::no;
    }

    virtual inline void get_bin(Bin& bin, const ProjDataInfo& proj_data_info) const
    {
        delta_timing_bin > 0 ?
                    bin.timing_pos_num() = static_cast<int> ( ( delta_timing_bin / proj_data_info.get_tof_mash_factor()) + 0.5)
                : bin.timing_pos_num() = static_cast<int> ( ( delta_timing_bin / proj_data_info.get_tof_mash_factor()) - 0.5);

        if (bin.timing_pos_num() <  proj_data_info.get_min_timing_pos_num() ||
                bin.timing_pos_num() > proj_data_info.get_max_timing_pos_num())
        {
            bin.set_bin_value(-1.f);
        }
    }

private:

    //!
    //! \brief timeA
    //! \details The detection time of the first of the two photons, in seconds
    float timeA;

    float delta_time;
};

//! A class for a general element of a listmode file for a Siemens scanner using the ROOT files
class CListRecordROOT : public CListRecord // currently no gating yet
{
public:
    //! Returns always true
    bool inline is_time() const;
    //! Returns always true
    bool inline is_event() const;
    //! Returns always true
    bool inline is_full_event() const;

    virtual CListEventROOT&  event()
    {
        return this->event_data;
    }

    virtual const CListEventROOT& event() const
    {
        return this->event_data;
    }

    virtual CListTimeROOT& time()
    {
        return this->time_data;
    }

    virtual const CListTimeROOT& time() const
    {
        return this->time_data;
    }

    bool operator==(const CListRecord& e2) const
    {
        return dynamic_cast<CListRecordROOT const *>(&e2) != 0 &&
                raw[0] == dynamic_cast<CListRecordROOT const &>(e2).raw[0] &&
                raw[1] == dynamic_cast<CListRecordROOT const &>(e2).raw[1];
    }

    CListRecordROOT(const shared_ptr<Scanner>& scanner_sptr) :
        event_data(scanner_sptr)
    {}

    virtual Succeeded init_from_data( const int& ring1,
                                      const int& ring2,
                                      const int& crystal1,
                                      const int& crystal2,
                                      float time1, float delta_time,
                                      const int& event1, const int& event2)
    {
        /// \warning ROOT data are time and event at the same time.

        this->event_data.init_from_data(ring1, ring2,
                                        crystal1, crystal2);

        if(!this->event_data.is_swapped())
            this->time_data.init_from_data(
                    time1, delta_timing_bin);
        else
        {
//            delta_timing_bin = -delta_timing_bin;
            this->time_data.init_from_data(
                        time1, -delta_timing_bin);
        }

        // We can make a singature raw based on the two events IDs.
        // It is pretty unique.
        raw[0] = event1;
        raw[1] = event2;

        return Succeeded::yes;
    }

private:
    CListEventROOT  event_data;
    CListTimeROOT   time_data;
    boost::int32_t raw[2]; // this raw field isn't strictly necessary, get rid of it?

};

END_NAMESPACE_STIR
#include "stir/listmode/CListRecordROOT.inl"
#endif

