// 
// $Id$
//
/*!
  \file
  \ingroup buildblock

  \brief This declares the Timer class.

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/

#ifndef __Tomo_TIMER_H__
#define __Tomo_TIMER_H__

#include "tomo/common.h"

START_NAMESPACE_TOMO

/*!
  \ingroup buildblock
  \brief
  A general base class for timers. Interface is like a stop watch.

  \par Example of usage:

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
   double get_current_value()
*/
class Timer
{  
public:
  inline Timer();
  inline virtual ~Timer();  
  inline void start();
  inline void stop();
  inline void reset();
#ifdef OLDDESIGN
  // TODO remove
  inline void restart() ;
#endif
  //! return value is undefined when start() is not called first.
  inline double value() const;

protected:
  bool running;
  double previous_value;
  double previous_total_value;
  
  virtual double get_current_value() const = 0;
};

END_NAMESPACE_TOMO


#include "Timer.inl"

// TODO remove
#ifdef OLDDESIGN
#include "CPUTimer.h"
#endif

#endif // __TIMER_H__
