//
// $Id$
//
/*!
  \file
  \ingroup buildblock
  \brief inline implementations for TimedObject

  \author Kris Thielemans
  \author PARAPET project

  $Date$

  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

START_NAMESPACE_STIR

void TimedObject::reset_timers()
{
  cpu_timer.reset();
}

void TimedObject::start_timers() const
{
  cpu_timer.start();
}


void TimedObject::stop_timers() const
{
  cpu_timer.stop();
}


double TimedObject::get_CPU_timer_value() const
{
  return cpu_timer.value();
}

END_NAMESPACE_STIR
