// @(#)Timer.h	1.6: 00/03/23
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/

/*!
 \file 
 \ingroup buildblock

 \brief Class stir::TimedBlock

 \author Alexey Zverovich
 \author PARAPET project

*/
#ifndef _stir_TimedBlock_H_
#define _stir_TimedBlock_H_
namespace stir {

  class Timer;

  /*! \brief Helper class for measuring execution time of a block of code.
  \ingroup buildblock

  It starts the timer in ctor, stops in dtor.
  Do not create unnamed instances of this class, as they are quite
  useless: you cannot predict destruction time.

  \par Usage:

  \code

  SomeTimer t;
  // do whatever you want here
  {
  TimedBlock<SomeTimer> tb(t);
  do_something_1();
  do_something_2();
  };
  // do whatever you want here
  {
  TimedBlock<SomeTimer> tb(t);
  do_something_3();
  };
  // do whatever you want here
  cout << "It took " << t.GetTime() << "sec to execute do_something_1..3()" << endl;

  \endcode

  \par Template argument requirements

  \c TimerT has to have a start() and stop() member function. This is the case for 
  stir::Timer (and derived functions) and stir::HighResWallClockTimer.
  */
  template <class TimerT>
  class TimedBlock<TimerT=Timer>
  {
  public:

    //! Create a timed block
    inline TimedBlock(TimerT& Timer);
    //! Destroy a timed block
    inline virtual ~TimedBlock(void);

  protected:
  private:

    TimedBlock(const TimedBlock&);            // Not defined
    TimedBlock& operator=(const TimedBlock&); // Not defined

    TimerT& m_Timer;

  };


  template <class TimerT>
    TimedBlock<TimerT>::TimedBlock(TimerT& timer)
    :   m_Timer(timer)
    {
      m_Timer.start();
    }

    template <class TimerT>
      TimedBlock<TimerT>::~TimedBlock(void)
      {
	m_Timer.stop();
      }
}

#endif
