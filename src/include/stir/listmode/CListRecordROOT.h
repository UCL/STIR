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
#include "stir/ProjDataInfoCylindrical.h"
#include "stir/Succeeded.h"
#include "stir/ByteOrder.h"
#include "stir/ByteOrderDefine.h"
#include "stir/round.h"
#include "boost/static_assert.hpp"
#include "boost/cstdint.hpp"
#include "stir/DetectionPositionPair.h"

START_NAMESPACE_STIR

//!
/*! \todo This implementation only works if the list-mode data is stored without axial compression.
  \todo If the target sinogram has the same characteristics as the sinogram encoding used in the list file
  (via the offset), the code could be sped-up dramatically by using the information.
  At present, we go a huge round-about (offset->sinogram->detectors->sinogram->offset)
*/
class CListEventROOT : public CListEventCylindricalScannerWithDiscreteDetectors
{
public:
//    typedef CListEventDataROOT DataType;
//    DataType get_data() const {
//        error("Cannor return datatype");
////        return this->data;
//    }

    CListEventROOT(const shared_ptr<ProjDataInfo>& proj_data_info_sptr);

    //! This routine returns the corresponding detector pair
    virtual void get_detection_position(DetectionPositionPair<>&) const;

    //! This routine sets in a coincidence event from detector "indices"
    virtual void set_detection_position(const DetectionPositionPair<>&);

    bool is_data() const
    { return true; }

    bool is_swapped() const
    { return swapped; }

    //!
    //! \brief init_from_data
    //! \param ring1
    //! \param ring2
    //! \param crystal1
    //! \param crystal2
    //! \return
    //! \details This is the main function which transform GATE coordinates to STIR
    //! scanner coordintates
    Succeeded init_from_data(int _ring1, int _ring2,
                             int crystal1, int crystal2)
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
        det1 = crystal1 + (int)(scanner_sptr->get_num_detectors_per_ring()/4.f);
        det2 = crystal2 + (int)(scanner_sptr->get_num_detectors_per_ring()/4.f);

        if  (det1 < 0 )
            det1 = scanner_sptr->get_num_detectors_per_ring() + det1;
        else if ( det1 >= scanner_sptr->get_num_detectors_per_ring())
            det1 = det1 - scanner_sptr->get_num_detectors_per_ring();

        if  (det2 < 0 )
            det2 = scanner_sptr->get_num_detectors_per_ring() + det2;
        else if ( det2 >= scanner_sptr->get_num_detectors_per_ring())
            det2 = det2 - scanner_sptr->get_num_detectors_per_ring();

        if (det1 > det2)
        {
            int tmp = det1;
            det1 = det2;
            det2 = tmp;

            ring1 = _ring2;
            ring2 = _ring1;
            swapped = true;
        }
        else
        {
            ring1 = _ring1;
            ring2 = _ring2;
            swapped = false;
        }

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
    int ring1;
    int ring2;

    int det1;
    int det2;

    bool swapped;

    std::vector<int> segment_sequence;
    std::vector<int> sizes;

};

//! A class for storing and using a timing 'event' from a listmode file from the ECAT 8_32bit scanner
/*! \ingroup listmode
 */
class CListTimeROOT : public CListTime
{
public:
    Succeeded init_from_data(double time1, double time2)
    {
        timeA = time1;
        timeB = time2;
        return Succeeded::yes;
    }

    bool is_time() const
    { return true; }

    //!
    //! \brief get_time_in_millisecs
    //! \return Returns the detection time of the first photon
    //! in milliseconds.
    //!
    inline unsigned long  get_time_in_millisecs() const
    {
        return timeA  * 1e3;
    }

    //!
    //! \brief get_timeA_in_milisecs
    //! \return Returns the detection time of the first photon
    //! in milliseconds
    //!
    inline double get_timeA_in_millisecs() const
    {
        return timeA * 1e3;
    }

    //!
    //! \brief get_timeB_in_milisecs
    //! \author Nikos Efthimiou
    //! \return Returns the detection time of the second photon
    //! in milliseconds
    //!
    inline double get_timeB_in_millisecs() const
    {
        return timeB * 1e3;
    }

    //!
    //! \brief get_delta_time_in_millisecs
    //! \author Nikos Efthimiou
    //! \return Returns the delta Time between the two events
    //!
    inline double get_delta_time_in_millisecs() const
    {
        return (timeB - timeA) * 1e3;
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
        return (timeB - timeA) * 1e12;
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
        error("set_time_in_millisecs: Not implemented for ROOT files. Abort.");
    }

private:

    //!
    //! \brief timeA
    //! \details The detection time of the first of the two photons, in seconds
    double timeA; // Detection time of the first event

    //!
    //! \brief timeB
    //! \details The detection time of the second of the two photons
    double timeB; // Detection time of the second event
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
                                      double time1, double time2)
    {
        /// \warning ROOT data are time and event at the same time.

        this->event_data.init_from_data(ring1, ring2, crystal1, crystal2);

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

