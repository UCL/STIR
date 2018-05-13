 
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
  \author Harry Tsoumpas
*/

#ifndef __stir_listmode_CListTimeROOT_H__
#define __stir_listmode_CListTimeROOT_H__

#include "stir/listmode/CListTime.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR


//! A class for storing and using a timing 'event' from a listmode file from the ECAT 8_32bit scanner
/*! \ingroup listmode
 */
class CListTimeROOT : public CListTime
{
public:
    void init_from_data(const double& time1, const double& time2)
    {
        timeA = time1;
        timeB = time2;
    }

    //! Returns always true
    inline bool is_time() const
    { return true; }
    //! Returns the detection time of the first photon
    //! in milliseconds.
    inline unsigned long  get_time_in_millisecs() const
    { return static_cast<unsigned long>(get_timeA_in_millisecs()); }
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
        warning("set_time_in_millisecs: Not implemented for ROOT files. Abort.");
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

END_NAMESPACE_STIR

#endif

