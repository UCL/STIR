//
// $Id$: $Date$
//

/*
  For most systems, CPUTimer::value() returns elapsed CPU time using the 
  ANSI clock() function.  From the man page: "The reported time is the sum
  of the CPU time of the calling process and its terminated child
  processes for which it has executed wait, system, or pclose
  subroutines."
  clock() has a possible problem of wrap-around. For many Unix systems this
  occurs very soon. For a SUN, the man page states this is every 2147 secs 
  of CPU time (36 minutes !). AIX should be similar, as both have 
  CLOCKS_PER_SEC = 10e6, and clock_t == int == 32 bit.

  So, since version 1.2 the times() function (with the rms_utime field) 
  is used #if defined(__OS_UNIX__). The  man page states something very similar 
  to clock():
  "tms_utime       The CPU time used for executing instructions in the user
   space of the calling process.

   This information is read from the calling process as well as from
   each completed child process for which the calling process executed
   a wait subroutine."
  As times() returns results measured in clock interrupt clicks (on AIX
  100 per second), wrap around occurs much later.


  Finally, VC++ 5.0 seems to have a bug in clock() that makes its results
  dependent on the load of the system (i.e. it is more like 'wall clock 
  time').
  We now use GetProcessTimes() (with the lpUserTime variable) instead. 
  The documentation states:
  "... the amount of time that the process has executed in user mode.
  The time that each of the threads of the process has executed in user 
  mode is determined, and then all of those times are summed together to 
  obtain this value. 
  Warning: this only works for NT, not for Win95

  History:
   1.0 by Kris Thielemans
   1.1 by Kris Thielemans
     use times() and GetProcessTimes()
   1.2 by Kris Thielemans
     moved inlines to separate file
*/

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

