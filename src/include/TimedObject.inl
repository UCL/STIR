//
// $Id$: $Date$
//
/*!
  \file
  \ingroup buildblock
  \brief inline implementations for TimedObject

  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/

START_NAMESPACE_TOMO

void TimedObject::reset_timers()
{
  cpu_timer.reset();
}

void TimedObject::start_timers()
{
  cpu_timer.start();
}


void TimedObject::stop_timers()
{
  cpu_timer.stop();
}


double TimedObject::get_CPU_timer_value() const
{
  return cpu_timer.value();
}

END_NAMESPACE_TOMO
