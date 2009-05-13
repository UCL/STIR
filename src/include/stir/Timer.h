// 
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
/*!
  \file
  \ingroup buildblock

  \brief This declares the stir::Timer class.

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
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
  inline void start();
  inline void stop();
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
