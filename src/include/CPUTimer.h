// 
// $Id$: $Date$
//
#ifndef __CPUTimer_H__
#define __CPUTimer_H__

#include "Timer.h"

/*

  CPUTimer is derived from Timer, and hence has the same interface. It
  returns the amount of CPU time (in secs) used by the current process and its 
  children. 
  Warning: If too long CPU times are measured, wrap-around will occur. 
  See CPUTimer.inl for details.

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

#include "CPUTimer.inl"

#endif // __CPUTimer_H__
