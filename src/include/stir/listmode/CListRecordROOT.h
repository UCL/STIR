/*
    Copyright (C) 2003-2011 Hammersmith Imanet Ltd
    Copyright (C) 2013-2014 University College London
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
#include "stir/ProjDataInfoCylindrical.h"
#include "stir/Succeeded.h"
#include "stir/ByteOrder.h"
#include "stir/ByteOrderDefine.h"
#include "stir/round.h"
#include "boost/static_assert.hpp"
#include "boost/cstdint.hpp"
#include "stir/DetectionPositionPair.h"

START_NAMESPACE_STIR

//! Class for decoding storing and using a raw coincidence event from a listmode file from any ROOT file
//! \todo More information could be usefull
class CListEventDataROOT
{
public:
    int ring1;
    int ring2;

    int det1;
    int det2;

    bool swap;

    bool random;
    bool scattered;
};

//!
/*! \todo This implementation only works if the list-mode data is stored without axial compression.
  \todo If the target sinogram has the same characteristics as the sinogram encoding used in the list file
  (via the offset), the code could be sped-up dramatically by using the information.
  At present, we go a huge round-about (offset->sinogram->detectors->sinogram->offset)
*/
class CListEventROOT : public CListEventCylindricalScannerWithDiscreteDetectors
{
public:
    typedef CListEventDataROOT DataType;
    DataType get_data() const {
        return this->data;
    }

    CListEventROOT(const shared_ptr<ProjDataInfo>& proj_data_info_sptr);

    //! This routine returns the corresponding detector pair
    //! \date  11/03/16
    //! \author Nikos Efthimiou
    //! \details I changed the return type is Succeeded.
    virtual Succeeded get_detection_position(DetectionPositionPair<>&) const;

    //! This routine sets in a coincidence event from detector "indices"
    //! \date  11/03/16
    //! \author Nikos Efthimiou
    //! \details I changed the return type is Succeeded.
    virtual Succeeded set_detection_position(const DetectionPositionPair<>&);

    bool is_data() const
    {
        return true;
    }

    bool is_random() const
    {
        return data.random;
    }

    bool is_scattered() const
    {
        return data.scattered;
    }

    bool is_swapped() const
    {
        return data.swap;
    }

    //!
    //! \brief init_from_data
    //! \param ring1
    //! \param ring2
    //! \param crystal1
    //! \param crystal2
    //! \param event1
    //! \param event2
    //! \param scattered1
    //! \param scattered2
    //! \return
    //! \author Nikos Efthimiou
    //! \details This is the main function which transform GATE coordinates to STIR
    //! scanner coordintates
    Succeeded init_from_data(int ring1, int ring2,
                             int crystal1, int crystal2,
                             int event1, int event2,
                             int scattered1, int scattered2)
    {


        if  (crystal1 < 0 )
            crystal1 = scanner_sptr->get_num_detectors_per_ring() + crystal1;
        else if ( crystal1 >= scanner_sptr->get_num_detectors_per_ring())
            crystal1 = crystal1 - scanner_sptr->get_num_detectors_per_ring();


        if  (crystal2 < 0 )
            crystal2 = scanner_sptr->get_num_detectors_per_ring() + crystal2;
        else if ( crystal2 >= scanner_sptr->get_num_detectors_per_ring())
            crystal2 = crystal2 - scanner_sptr->get_num_detectors_per_ring();

        // STIR assumes that 0 is on y whill GATE on the x axis
        data.det1 = crystal1 + (int)(scanner_sptr->get_num_detectors_per_ring()/4)  ;
        data.det2 = crystal2  + (int)(scanner_sptr->get_num_detectors_per_ring()/4) ;

        if  (data.det1 < 0 )
            data.det1 = scanner_sptr->get_num_detectors_per_ring() + data.det1;
        else if ( data.det1 >= scanner_sptr->get_num_detectors_per_ring())
            data.det1 = data.det1 - scanner_sptr->get_num_detectors_per_ring();

        if  (data.det2 < 0 )
            data.det2 = scanner_sptr->get_num_detectors_per_ring() + data.det2;
        else if ( data.det2 >= scanner_sptr->get_num_detectors_per_ring())
            data.det2 = data.det2 - scanner_sptr->get_num_detectors_per_ring();

        if (data.det1 > data.det2)
        {
            int tmp = data.det1;
            data.det1 = data.det2;
            data.det2 = tmp;

            data.ring1 = ring2;
            data.ring2 = ring1;
        }
        else
        {
            data.ring1 = ring1;
            data.ring2 = ring2;
        }

        if (event1 != event2)
            data.random = true;
        else
            data.random = false;

        if (scattered1 > 0 || scattered2 > 0)
            data.scattered = true;
        else
            data.scattered = false;

        return Succeeded::yes;
    }

    //!
    //! \brief is_prompt
    //! \return Returns always true
    //! \author Nikos Efthimiou
    //! \warning All events are prompt, because in order to store in
    //! GATE delayed events, a new coincidence chain has to be used.
    //!
    inline bool is_prompt() const
    {
        return true;
    }

    inline Succeeded set_prompt(const bool prompt = true)
    {
        //        if (prompt) this->data.delayed=1; else this->data.delayed=0; return Succeeded::yes;
    }

private:

    CListEventDataROOT   data;


    std::vector<int> segment_sequence;
    std::vector<int> sizes;

};

//!
//! \brief The CListTimeDataROOT class
//! \author Nikos Efthimiou
//! \details This is the core class to hold the timing data of the event.
//! In order to provide potential TOF compatibility CListTimeDataROOT holds
//! the time stamps in picosecods.
//!
class CListTimeDataROOT
{
public:

    //!
    //! \brief timeA
    //! \author Nikos Efthimiou
    //! \details The detection time of the first of the two photons, in seconds
    double timeA; // Detection time of the first event

    //!
    //! \brief timeB
    //! \author Nikos Efthimiou
    //! \details The detection time of the second of the two photons
    double timeB; // Detection time of the second event

    //!
    //! \brief type
    //! \warning No actual functionality. I keep it for compatibility with
    //! ECAT data
    unsigned    type : 1;    /* 0-coincidence event, 1-time tick */
};


//! A class for storing and using a timing 'event' from a listmode file from the ECAT 8_32bit scanner
/*! \ingroup listmode
 */
class CListTimeROOT : public CListTime
{
public:
    Succeeded init_from_data(double time1, double time2)
    {

        data.timeA = time1;
        data.timeB = time2;

        return Succeeded::yes;
    }

    bool is_time() const
    {
        return true;
    }

    //!
    //! \brief get_time_in_millisecs
    //! \return Returns the detection time of the first photon
    //! in milliseconds.
    //!
    inline unsigned long  get_time_in_millisecs() const
    {
        return this->data.timeA  * 1e3;
    }

    //!
    //! \brief get_timeA_in_milisecs
    //! \return Returns the detection time of the first photon
    //! in milliseconds
    //!
    inline double get_timeA_in_millisecs() const
    {
        return this->data.timeA * 1e3;
    }

    //!
    //! \brief get_timeB_in_milisecs
    //! \author Nikos Efthimiou
    //! \return Returns the detection time of the second photon
    //! in milliseconds
    //!
    inline double get_timeB_in_millisecs() const
    {
        return this->data.timeB * 1e3;
    }

    //!
    //! \brief get_delta_time_in_millisecs
    //! \author Nikos Efthimiou
    //! \return Returns the delta Time between the two events
    //!
    inline double get_delta_time_in_millisecs() const
    {
        return (this->data.timeB - this->data.timeA) * 1e3;
    }

    //!
    //! \brief get_delta_time_in_picosecs
    //! \return
    //! \author Nikos Efthimiou
    //! \details Since I chose to use int numbers as tof info containers,
    //! in the Bin class, this could be usefull.
    //!
    inline  double get_delta_time_in_picosecs() const
    {
        double delta = (this->data.timeB - this->data.timeA) ;
        delta *= 1e12;
        return delta;
    }

    //!
    //! \brief set_time_in_millisecs
    //! \param time_in_millisecs
    //! \return
    //! \warning I removed all functionality and probably should be deleted in
    //! a future version
    //!
    inline Succeeded set_time_in_millisecs(const unsigned long time_in_millisecs)
    {
        //        this->data.time = ((1U<<30)-1) & static_cast<unsigned>(time_in_millisecs);
        //        // TODO return more useful value
        //        return Succeeded::yes;
    }

private:
    CListTimeDataROOT   data;
};

//! A class for a general element of a listmode file for a Siemens scanner using the ROOT files
class CListRecordROOT : public CListRecord // currently no gating yet
{

    //public:

    bool is_time() const
    {
        return this->time_data.is_time();
    }
    /*
  bool is_gating_input() const
  { return this->is_time(); }
  */
    bool is_event() const
    {
        return this->event_data.is_data();
    }

    bool is_random() const
    {
        return this->event_data.is_random();
    }

    bool is_scattered() const
    {
        return this->event_data.is_scattered();
    }

    bool is_full_event()
    {
        return true;
    }

    virtual CListEventROOT&  event()
    {
        return this->event_data;
    }

    virtual const CListEventROOT&  event() const
    {
        return this->event_data;
    }

    virtual CListTimeROOT&   time()
    {
        return this->time_data;
    }

    virtual const CListTimeROOT&   time() const
    {
        return this->time_data;
    }

    bool operator==(const CListRecord& e2) const
    {
        return dynamic_cast<CListRecordROOT const *>(&e2) != 0 &&
                raw == dynamic_cast<CListRecordROOT const &>(e2).raw;
    }

public:
    CListRecordROOT(const shared_ptr<ProjDataInfo>& proj_data_info_sptr) :
        event_data(proj_data_info_sptr)
    {}

    virtual Succeeded init_from_data( int ring1, int ring2,
                                      int crystal1, int crystal2,
                                      double time1, double time2,
                                      int event1, int event2,
                                      int scattered1, int scattered2)
    {
        /// \warning ROOT data are time and event at the same time.

        this->event_data.init_from_data(ring1, ring2, crystal1, crystal2,
                                        event1, event2,
                                        scattered1, scattered2);

        this->time_data.init_from_data(
                    time1,time2);

        return Succeeded::yes;
    }

    virtual std::size_t size_of_record_at_ptr(const char * const /*data_ptr*/, const std::size_t /*size*/,
                                              const bool /*do_byte_swap*/) const
    { return 4; }

private:
    CListEventROOT  event_data;
    CListTimeROOT   time_data;
    boost::int32_t         raw; // this raw field isn't strictly necessary, get rid of it?

};

END_NAMESPACE_STIR

#endif

