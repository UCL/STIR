// 
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    Copyright (C) 2023, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup buildblock

  \brief This declares the stir::Timer class.

  \author Kris Thielemans
  \author PARAPET project

*/

#ifndef __stir_TIMER_H__
#define __stir_TIMER_H__

#include "stir/common.h"

START_NAMESPACE_STIR

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
  //! start stopwatch, optionally resetting first
  /*! the stopwatch should not be running already if asking to reset */
  inline void start(bool do_reset = false);
  //! stop stopwatch
  inline void stop();
  //! reset stopwatch
  /*! the stopwatch should not be running */
  inline void reset();
  //! return value is undefined when start() is not called first.
  inline double value() const;

protected:
  bool running;
  double previous_value;
  double previous_total_value;
  
  virtual double get_current_value() const = 0;
};

END_NAMESPACE_STIR


#include "stir/Timer.inl"

#endif // __TIMER_H__
