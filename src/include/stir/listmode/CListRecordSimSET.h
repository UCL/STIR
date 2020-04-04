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

#include "stir/listmode/CListRecord.h"
#include "stir/listmode/CListEventCylindricalScannerWithDiscreteDetectors.h"
#include "stir/Succeeded.h"

extern "C" {
#include <LbTypes.h>
#include <Photon.h>
}

START_NAMESPACE_STIR

class CListEventSimSET : public CListEventCylindricalScannerWithDiscreteDetectors
{
public:

    CListEventSimSET(const shared_ptr<ProjDataInfo>& proj_data_info);

    //! This routine returns the corresponding detector pair
    virtual void get_detection_position(DetectionPositionPair<>&) const;

    //! This routine sets in a coincidence event from detector "indices"
    virtual void set_detection_position(const DetectionPositionPair<>&);

    //! \details This is the main function which transform GATE coordinates to STIR
    void init_from_data(const PHG_DetectedPhoton* _blue,
                        const PHG_DetectedPhoton* _pink,
                        const float _weight,
                        const float _tofDifference = 0.0);

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
    //! TOF time difference between the two photons
    float tofDifference;

};

//! A class for storing and using a timing 'event' from a listmode file from the ECAT 8_32bit scanner
/*! \ingroup listmode
 */
class CListTimeSimSET : public CListTime
{
public:
    inline void init_from_data(const PHG_DetectedPhoton* _blue,
                               const PHG_DetectedPhoton* _pink)
    {
        timeA = _blue->time_since_creation;
        timeB = _pink->time_since_creation;
    }

    //! Returns always true
    inline bool is_time() const
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
        warning("set_time_in_millisecs: Not implemented for SimSET files. Aborting.");
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
class CListRecordSimSET : public CListRecord // currently no gating yet
{
public:
    //! Returns always true
    bool inline is_time() const;
    //! Returns always true
    bool inline is_event() const;
    //! Returns always true
    bool inline is_full_event() const;

    virtual CListEventSimSET&  event()
    {
        return this->event_data;
    }

    virtual const CListEventSimSET& event() const
    {
        return this->event_data;
    }

    virtual CListTimeSimSET& time()
    {
        return this->time_data;
    }

    virtual const CListTimeSimSET& time() const
    {
        return this->time_data;
    }

    bool operator==(const CListRecord& e2) const
    {
        return this->event().get_LOR().p1() ==  e2.event().get_LOR().p1() &&
                this->event().get_LOR().p2() ==  e2.event().get_LOR().p2() &&
                this->time().get_time_in_secs() == this->time().get_time_in_secs();
    }

    CListRecordSimSET(const shared_ptr<ProjDataInfo>& proj_data_info) :
        event_data(proj_data_info)
    {}

    virtual Succeeded init_from_data(const PHG_DetectedPhoton&	detectedPhotonBlue,
                                     const PHG_DetectedPhoton&	detectedPhotonPink,
                                     const float weight,
                                     const float tofDifference)
    {
        this->event_data.init_from_data(&detectedPhotonBlue,
                                        &detectedPhotonPink,
                                        weight,
                                        tofDifference);

        this->time_data.init_from_data(&detectedPhotonBlue,
                                       &detectedPhotonPink);

        return Succeeded::yes;
    }

private:
    CListEventSimSET  event_data;
    CListTimeSimSET   time_data;
};

END_NAMESPACE_STIR
#include "stir/listmode/CListRecordSimSET.inl"
#endif

