#ifndef __TimedObject_H__
#define __TimedObject_H__
/*!
  \file 
  \ingroup buildblock
 
  \brief declares the stir::TimedObject class

  \author Kris Thielemans  
  \author PARAPET project

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
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

#include "stir/CPUTimer.h"

START_NAMESPACE_STIR
/*!
  \ingroup buildblock
  \brief base class for all objects which need timers. 
  At the moment, there's only a CPU timer.

  It is the responsibility of the derived class to start and 
  stop the timer.
*/
class TimedObject 
{
public:

  //! reset all timers kept by this object
  inline void reset_timers();

  //! stop all timers kept by this object
  /*! Note: function is const such that it can be called in a
      function of a derived class that is const.
  */
  inline void stop_timers() const;

  //! start all timers kept by this object
  /*! Note: function is const such that it can be called in a
      function of a derived class that is const.
  */
  inline void start_timers() const;

  //! get current value of the timer (since first use or last reset)
  inline double get_CPU_timer_value() const;

private:

  //! A timer that measured CPU time.
  /*! Note: member is mutable such that it can be modified in a const function.
  */
  mutable CPUTimer cpu_timer;

  // TODO include other times (such as wall-clock)

};

END_NAMESPACE_STIR

#include "stir/TimedObject.inl"

#endif
