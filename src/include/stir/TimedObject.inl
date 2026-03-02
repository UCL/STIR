//
//
/*!
  \file
  \ingroup buildblock
  \brief inline implementations for stir::TimedObject

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
