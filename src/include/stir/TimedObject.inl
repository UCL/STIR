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

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

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
