//
// $Id$: $Date$
//
/*!
  \file 
 
  \brief inline implementations for CPUTimer

  \author Kris Thielemans
  \author PARAPET project

  \date    $Date$

  \version $Revision$

*/

/*
  History:
   1.0 by Kris Thielemans
   1.1 by Kris Thielemans
     use times() and GetProcessTimes()
   1.2 by Kris Thielemans
     moved inlines to separate file
*/
START_NAMESPACE_TOMO

// KT 12/01/2000 use for all unixes
#if defined(__OS_UNIX__)
// use times() instead of clock() for Unix. (Higher resolution)
// Only tested for AIX, sun, OSF, but it is probably POSIX
// If does not work for your OS, use the version with clock() below

#include <sys/times.h>
#include <unistd.h>

double CPUTimer::get_current_value() const
{  
  struct tms t;
  times(&t);
  return double( t.tms_utime ) / sysconf(_SC_CLK_TCK);
}


// KT 12/01/2000 use on every windows system
#elif defined(__OS_WIN__)

// KT 27/05/98 VC++ 5.0 needs GetProcessTimes(), due to a bug in clock()
// This breaks on Win95, but should work on WinNT
#include <windows.h>
double CPUTimer::get_current_value() const
{  
  FILETIME CreationTime;  // when the process was created 
  FILETIME ExitTime;  // when the process exited 
  FILETIME KernelTime;  // time the process has spent in kernel mode 
  FILETIME UserTime;  // time the process has spent in user mode 
  
  GetProcessTimes(GetCurrentProcess(), 
    &CreationTime, &ExitTime, &KernelTime, &UserTime);
  // KT 13/01/2000 gcc still has problems with LARGE_INTEGER
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

double CPUTimer::get_current_value() const
{  
  return double( clock() ) / CLOCKS_PER_SEC;
}
#endif


END_NAMESPACE_TOMO
