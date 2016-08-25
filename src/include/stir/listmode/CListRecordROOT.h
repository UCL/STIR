/*
    Copyright (C) 2015-2016 University of Leeds
    Copyright (C) 2016 UCL
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
*/

#ifndef __stir_listmode_CListRecordROOT_H__
#define __stir_listmode_CListRecordROOT_H__

#include "stir/listmode/CListRecord.h"
#include "stir/listmode/CListEventCylindricalScannerWithDiscreteDetectors.h"
#include "stir/Succeeded.h"
#include "stir/round.h"
#include "boost/static_assert.hpp"
#include "stir/DetectionPositionPair.h"

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
    void init_from_data(double time1, double time2)
    {
        timeA = time1;
        timeB = time2;
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
    inline double get_timeA_in_millisecs() const
    { return timeA * 1e3; }
    //! Get the detection time of the second photon
    //! in milliseconds
    inline double get_timeB_in_millisecs() const
    { return timeB * 1e3; }
    //! Get the delta Time between the two events
    inline double get_delta_time_in_millisecs() const
    { return (timeB - timeA) * 1e3; }
    //! Get delta time in picoseconds
    inline  double get_delta_time_in_picosecs() const
    { return (timeB - timeA) * 1e12; }
    inline Succeeded set_time_in_millisecs(const unsigned long time_in_millisecs)
    {
        warning("set_time_in_millisecs: Not implemented for ROOT files. Aborting.");
        return Succeeded::no;
    }

private:

    //!
    //! \brief timeA
    //! \details The detection time of the first of the two photons, in seconds
    double timeA;

    //!
    //! \brief timeB
    //! \details The detection time of the second of the two photons
    double timeB;
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
                raw == dynamic_cast<CListRecordROOT const &>(e2).raw;
    }

    CListRecordROOT(const shared_ptr<Scanner>& scanner_sptr) :
        event_data(scanner_sptr)
    {}

    virtual Succeeded init_from_data( const int& ring1,
                                      const int& ring2,
                                      const int& crystal1,
                                      const int& crystal2,
                                      double time1, double time2,
                                      const int& event1, const int& event2)
    {
        /// \warning ROOT data are time and event at the same time.

        this->event_data.init_from_data(ring1, ring2,
                                        crystal1, crystal2);

        this->time_data.init_from_data(
                    time1,time2);

        // We can make a singature raw based on the two events IDs.
        // It is pretty unique.
        raw = static_cast<boost::int64_t> (event1) << 32 | event2;

        return Succeeded::yes;
    }

private:
    CListEventROOT  event_data;
    CListTimeROOT   time_data;
    boost::int64_t         raw; // this raw field isn't strictly necessary, get rid of it?

};

END_NAMESPACE_STIR
#include "stir/listmode/CListRecordROOT.inl"
#endif

