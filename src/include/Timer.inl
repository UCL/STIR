//
// $Id$: $Date$
//
/*!

  \file
  \ingroup buildblock

  \brief inline implementations for Timer

  \author Kris Thielemans
  \author PARAPET project

  \date $Date$
  \version $Revision$
*/
START_NAMESPACE_TOMO

Timer::Timer()
{  
  running = false;
  previous_total_value = 0.;
  
}

Timer::~Timer()
{}

void Timer::start() 
{ 
  if (!running)
  {
    running = true;
    previous_value = get_current_value();
  }
}

void Timer::stop()
{ 
  if (running)
  {
    previous_total_value += get_current_value() - previous_value;
    running = false;
  }
}

#ifdef OLDDESIGN
void Timer::restart() 
{ 
  previous_total_value = 0.;
  running = false;
  start();
}
#endif

void Timer::reset() 
{ 
  assert(running == false);
  previous_total_value = 0.;
}

double Timer::value() const
{ 
  double tmp = previous_total_value;  
  if (running)
    tmp += get_current_value() - previous_value;
  return tmp;
}


END_NAMESPACE_TOMO
