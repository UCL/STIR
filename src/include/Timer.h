// 
// $Id$: $Date$
//
/*!

  \file

  \brief This declares the Timer class.

  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/

#ifndef __TIMER_H__
#define __TIMER_H__

#include "Tomography_common.h"

START_NAMESPACE_TOMO

/*!
  \brief
  a general base class for timers. Interface is like a stop watch.

  Example of usage:

  \code
  DerivedFromTimer t;
  t.start();
  // do things
  cerr << t.value();
  // do things
  t.stop();
  // do things
  t.reset()
  // etc
  \endcode

  Derived classes simply have to implement the virtual function 
  \c double get_current_value()
*/



class Timer
{  
public:
  inline Timer();
  inline virtual ~Timer();  
  inline void start();
  inline void stop();
  inline void reset();
  // TODO remove
  inline void restart() ;
  //! return value is undefined when start() is not called first.
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

END_NAMESPACE_TOMO

#endif // __TIMER_H__
