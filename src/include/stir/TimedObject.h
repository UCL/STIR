// $Id$: $Date$
#ifndef __TimedObject_H__
#define __TimedObject_H__
/*!
  \file 
  \ingroup buildblock
 
  \brief declares the TimedObject class

  \author Kris Thielemans  
  \author PARAPET project

  \date    $Date$
  \version $Revision$
*/

#include "CPUTimer.h"

START_NAMESPACE_TOMO
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
  inline void stop_timers();

  //! start all timers kept by this object
  inline void start_timers();

  //! get current value of the timer (since first use or last reset)
  inline double get_CPU_timer_value() const;

private:

  CPUTimer cpu_timer;

};

END_NAMESPACE_TOMO

#include "TimedObject.inl"

#endif
