// 
// $Id$: $Date$
//

/*
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

  History:
   1.0 by Kris Thielemans
   1.1 by Kris Thielemans
     CPUTimer uses times() and GetProcessTimes()
   1.2 by Kris Thielemans
     moved inlines to separate file
     made Timer::~Timer virtual

*/


#ifndef __TIMER_H__
#define __TIMER_H__

#include "pet_common.h"

class Timer
{  
public:
  inline Timer();
  // KT 12/01/2000 made virtual
  inline virtual ~Timer();  
  inline void start();
  inline void stop();
  inline void restart() ;
  inline double value() const;

protected:
  bool running;
  double previous_value;
  double previous_total_value;
  
  virtual double get_current_value() const = 0;
};

#include "Timer.inl"

// TODO remove
#include "CPUTimer.h"

#endif // __TIMER_H__


