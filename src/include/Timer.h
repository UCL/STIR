// $Id$: $Date$

/*
  This file defines two classes: Timer and a derived class CPUTimer.
  Version 1.0 and 1.1 by Kris Thielemans


  Timer is a general base class for timers. Interface is like a stop watch:

  DerivedFromTimer t;
  t.start();
  // do things
  cerr << t.value();
  // do things
  t.stop();
  // do things
  t.restart()
  // etc

  Derived classes simply have to implement the virtual function 
  double get_current_value()

  CPUTimer is derived from Timer, and hence has the same interface. It
  returns the amount of CPU time used by the current process and its 
  children. If too long CPU times are measured, there will occur a
  wrap-around. See below for details.

  For most systems, CPUTimer::value() returns elapsed CPU time using the 
  ANSI clock() function.  From the man page: "The reported time is the sum
  of the CPU time of the calling process and its terminated child
  processes for which it has executed wait, system, or pclose
  subroutines."
  clock() has a possible problem of wrap-around. For many Unix systems this
  occurs very soon. For a SUN, the man page states this is every 2147 secs 
  of CPU time (36 minutes !). AIX should be similar, as both have 
  CLOCKS_PER_SEC = 10e6, and clock_t == int == 32 bit.

  So, since version 1.1 the times() function (with the rms_utime field) 
  is used on AIX, SUN, OSF. The  man page states something very similar 
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
  We now uses GetProcessTimes() (with the lpUserTime variable) instead. 
  The documentation states:
  "... the amount of time that the process has executed in user mode.
  The time that each of the threads of the process has executed in user 
  mode is determined, and then all of those times are summed together to 
  obtain this value. 
  Warning: the VC++ doc says that this only works for NT.
*/



class Timer
{
protected:
  bool running;
  double previous_value;
  double previous_total_value;
  
  virtual double get_current_value() const = 0;
  
public:
  Timer()
    {  
      //restart(); 
      running = false;
      previous_total_value = 0.;
      
    }
  ~Timer()
    {}
    
  void start() 
    { 
      if (!running)
      {
        running = true;
        previous_value = get_current_value();
      }
    }

  void stop()
    { 
      if (running)
        {
          previous_total_value += get_current_value() - previous_value;
          running = false;
        }
    }

  void restart() 
    { 
      previous_total_value = 0.;
      running = false;
      start();
    }

  double value() const
    { 
      double tmp = previous_total_value;  
      if (running)
         tmp += get_current_value() - previous_value;
      return tmp;
    }
};

#if defined(_AIX) || defined (sun) || defined(__osf__)
// KT 27/05/98 use times() instead of clock() for AIX, sun, OSF.
// This probably works on other Unixs as well, but I couldn't check that.

#include <sys/times.h>
#include <unistd.h>

class CPUTimer : public Timer
{
private:
  virtual double get_current_value() const
    {  
      struct tms t;
      times(&t);
      return double( t.tms_utime ) / sysconf(_SC_CLK_TCK);
    }
};

#elif defined(_MSC_VER) && defined(_WIN32)
// KT 27/05/98 VC++ 5.0 needs GetProcessTimes(), due to a bug in clock()
// This possible breaks on Win95, but should work on WinNT
#include <windows.h>

class CPUTimer : public Timer
{
private:
  virtual double get_current_value() const
    {  
      FILETIME CreationTime;  // when the process was created 
      FILETIME ExitTime;  // when the process exited 
      FILETIME KernelTime;  // time the process has spent in kernel mode 
      FILETIME UserTime;  // time the process has spent in user mode 
      LARGE_INTEGER ll;

      GetProcessTimes(GetCurrentProcess(), 
		      &CreationTime, &ExitTime, &KernelTime, &UserTime);

      ll.LowPart = UserTime.dwLowDateTime;
      ll.HighPart =  UserTime.dwHighDateTime; 

      return double(ll.QuadPart ) / 1E7;
    }
};
#else // all other systems

#include <time.h>

class CPUTimer : public Timer
{
private:
  virtual double get_current_value() const
    {  
      return double( clock() ) / CLOCKS_PER_SEC;
    }
};

#endif
