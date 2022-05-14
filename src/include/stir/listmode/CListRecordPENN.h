/*
    Copyright (C) 2020-2022 University of Pennsylvania
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup listmode
  \brief Classes for listmode events for the PENNPET Explorer scanner.

  \author Nikos Efthimiou
*/

#ifndef __stir_listmode_CListRecordPENN_H__
#define __stir_listmode_CListRecordPENN_H__

#include "stir/listmode/CListEventCylindricalScannerWithDiscreteDetectors.h"

START_NAMESPACE_STIR

/*!
  \brief Class for handling PENNPet Explorer events.
  \ingroup listmode

  \todo Fix the rotation
  */
class CListEventPENN : public CListEventCylindricalScannerWithDiscreteDetectors
{
public:
    CListEventPENN(const shared_ptr<Scanner>& scanner_sptr);
    //! This routine returns the corresponding detector pair
    virtual void get_detection_position(DetectionPositionPair<>&) const;
    //! This routine sets in a coincidence event from detector "indices"
    virtual void set_detection_position(const DetectionPositionPair<>&);

    void init_from_data(bool d, int dt, int xa, int xb, int za, int zb, int ea, int eb);

    inline bool is_prompt() const
    { return delay == false; }

    inline Succeeded set_prompt(const bool _prompt = true)
    {
        if (_prompt)
            delay=false;
        else
            delay=true;
        return Succeeded::yes;
    }

private:

    std::vector<int> segment_sequence;
    std::vector<int> sizes;

    int delay;
#ifdef STIR_TOF
    short int tof_bin;
    short int orig_tof_bin;
#endif
    int d1, d2;
    int z1, z2;
#if 0
    // Most likely, most people will not need these.
    unsigned short int orig_z1, orig_z2;
#endif
    int quarter_of_detectors;

};

/*! \ingroup listmode
 */
class CListTimePENN : public ListTime
{
public:
    Succeeded init_from_data_ptr(const void * const ptr)
    {
        return Succeeded::yes;
    }
    bool is_time() const
    { return false; }
    inline unsigned long get_time_in_millisecs() const
    { return 0;  }
    inline Succeeded set_time_in_millisecs(const unsigned long time_in_millisecs)
    {
        return Succeeded::yes;
    }

private:

};

/*! \ingroup listmode
 */
class CListRecordPENN : public CListRecord // currently no gating yet
{

public:
    //!Currently, I skip time events.
    bool is_time() const
    { return false; }
    /*
  bool is_gating_input() const
  { return this->is_time(); }
  */
    bool is_event() const
    { return true; }
    virtual CListEventPENN&  event()
    { return this->event_data; }
    virtual const CListEventPENN&  event() const
    { return this->event_data; }
    virtual CListTimePENN&   time()
    { return this->time_data; }
    virtual const CListTimePENN&   time() const
    { return this->time_data; }
    //! \todo
    bool operator==(const CListRecord& e2) const
    {

    }

public:
    CListRecordPENN(const shared_ptr<Scanner>& scanner_sptr) :
        event_data(scanner_sptr)
    {}

    virtual Succeeded init_from_data(int is_delay,
                                     int dt,
                                     int xa, int xb,
                                     int za, int zb,
                                     int ea, int eb)
    {

        this->event_data.init_from_data(is_delay, dt,
                                        xa, xb,
                                        za, zb,
                                        ea, eb);
        //         this->time_data.init_from_data(ta, tb);

        return Succeeded::yes;
    }

private:
    CListEventPENN  event_data;
    CListTimePENN   time_data;
};

END_NAMESPACE_STIR

#endif

