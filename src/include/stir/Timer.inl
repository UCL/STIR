//
//
/*!

  \file
  \ingroup buildblock

  \brief inline implementations for stir::Timer

  \author Kris Thielemans
  \author PARAPET project

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
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
START_NAMESPACE_STIR

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


END_NAMESPACE_STIR
