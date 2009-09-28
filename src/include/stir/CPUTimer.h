// 
// $Id$
//
/*!

  \file
  \ingroup buildblock

  \brief declares the stir::CPUTimer class.
    
  \author Kris Thielemans
  \author PARAPET project
      
  $Date$        
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
#ifndef __CPUTimer_H__
#define __CPUTimer_H__

#include "stir/Timer.h"
START_NAMESPACE_STIR

/*!
  \ingroup buildblock
  \brief A class for measuring elapsed CPU time.

  CPUTimer is derived from Timer, and hence has the same interface. It
  returns the amount of CPU time (in secs) used by the current process and its 
  children. 
  \warning If too long CPU times are measured, wrap-around will occur. 
  \warning Currently, CPUTimer returns garbage on Win95 (and probably Win98). It does
  work fine on NT, XP etc though.
  
  \par Implementation details

  For most systems, CPUTimer::value() returns elapsed CPU time using the 
  ANSI clock() function.  From the man page: 
  \verbatim
  The reported time is the sum
  of the CPU time of the calling process and its terminated child
  processes for which it has executed wait, system, or pclose
  subroutines.
  \endverbatim
  clock() has a possible problem of wrap-around. For many Unix systems this
  occurs very soon. For a SUN, the man page states this is every 2147 secs 
  of CPU time (36 minutes !). AIX should be similar, as both have 
  CLOCKS_PER_SEC = 10e6, and clock_t == int == 32 bit.

  So, since version 1.2 the times() function (with the rms_utime field) 
  is used if <code>defined(__OS_UNIX__)</code>. The  man page for times() states something 
  very similar to clock():

  <i>
  tms_utime : The CPU time used for executing instructions in the user
   space of the calling process.

   This information is read from the calling process as well as from
   each completed child process for which the calling process executed
   a wait subroutine.
  </i>

  As times() returns results measured in clock interrupt clicks (on AIX
  100 per second), wrap around occurs much later.


  Finally, VC++ 5.0 seems to have a bug in clock() that makes its results
  dependent on the load of the system (i.e. it is more like 'wall clock 
  time').
  We now use GetProcessTimes() (with the lpUserTime variable) instead. 
  The documentation states:
  
  <i>
  ... the amount of time that the process has executed in user mode.
  The time that each of the threads of the process has executed in user 
  mode is determined, and then all of those times are summed together to 
  obtain this value. 
  </i>

  Warning: this only works for NT, not for Win95
*/
// TODO do Win95 test
/*
  History:
   1.0 by Kris Thielemans
   1.1 by Kris Thielemans
     use times() and GetProcessTimes()
   1.2 by Kris Thielemans
     moved inlines to separate file
*/
class CPUTimer : public Timer
{
private:
  virtual inline double get_current_value() const;
};



END_NAMESPACE_STIR

#include "stir/CPUTimer.inl"

#endif // __CPUTimer_H__

