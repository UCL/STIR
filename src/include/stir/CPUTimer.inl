//
//
/*!
  \file 
  \ingroup buildblock
 
  \brief inline implementations for stir::CPUTimer

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

/*
  History:
   1.0 by Kris Thielemans
   1.1 by Kris Thielemans
     use times() and GetProcessTimes()
   1.2 by Kris Thielemans
     moved inlines to separate file
*/


#if defined(__OS_UNIX__) && !defined(STIR_CPUTimer_use_clock)
// use times() instead of clock() for Unix. (Higher resolution)
// Only tested for AIX, sun, OSF, but it is probably POSIX
// If does not work for your OS, use the version with clock() below

#include <sys/times.h>
#include <unistd.h>

START_NAMESPACE_STIR

double CPUTimer::get_current_value() const
{  
  struct tms t;
  times(&t);
  return double( t.tms_utime ) / sysconf(_SC_CLK_TCK);
}



#elif defined(__OS_WIN__)  && !defined(STIR_CPUTimer_use_clock)

// VC++ 5.0 needs GetProcessTimes(), due to a bug in clock()
// This breaks on Win95 though
#include <windows.h>

START_NAMESPACE_STIR

// undefine the min,max macros again (we did this already in Tomography_common.h)
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif


double CPUTimer::get_current_value() const
{  
  FILETIME CreationTime;  // when the process was created 
  FILETIME ExitTime;  // when the process exited 
  FILETIME KernelTime;  // time the process has spent in kernel mode 
  FILETIME UserTime;  // time the process has spent in user mode 
  
  GetProcessTimes(GetCurrentProcess(), 
    &CreationTime, &ExitTime, &KernelTime, &UserTime);
  //  gcc still has problems with LARGE_INTEGER
#ifndef __GNUG__  
  LARGE_INTEGER ll;
  ll.LowPart = UserTime.dwLowDateTime;
  ll.HighPart =  UserTime.dwHighDateTime; 
  
  return static_cast<double>(ll.QuadPart ) / 1E7;
#else
  // can't use LONGLONG in cygwin B20.1
  const long long value = 
    (static_cast<long long>(UserTime.dwHighDateTime) << 32)
    + UserTime.dwLowDateTime;
  return static_cast<double>(value)/ 1E7;
#endif
}

#else // all other systems

#include <time.h>

START_NAMESPACE_STIR

double CPUTimer::get_current_value() const
{  
  return double( clock() ) / CLOCKS_PER_SEC;
}
#endif


END_NAMESPACE_STIR
