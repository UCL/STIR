// @(#)Timer.h	1.6: 00/03/23
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

 \brief Class stir::TimedBlock

 \author Alexey Zverovich
 \author PARAPET project

  $Date$
  $Revision$
*/
#ifndef _stir_TimedBlock_H_
#define _stir_TimedBlock_H_
namespace stir {

  /*! \brief Helper class for measuring execution time of a block of code.

  It starts the timer in ctor, stops in dtor.
  Do not create unnamed instances of this class, as they are quite
  useless: you cannot predict destruction time.

  \par Usage:

  \code

  SomeTimer t;
  // do whatever you want here
  {
  TimedBlock tb(t);
  do_something_1();
  do_something_2();
  };
  // do whatever you want here
  {
  TimedBlock tb(t);
  do_something_3();
  };
  // do whatever you want here
  cout << "It took " << t.GetTime() << "sec to execute do_something_1..3()" << endl;

  \endcode
  */

  class TimedBlock
  {
  public:

    //! Create a timed block
    TimedBlock(Timer& Timer);
    //! Destroy a timed block
    virtual ~TimedBlock(void);

  protected:
  private:

    TimedBlock(const TimedBlock&);            // Not defined
    TimedBlock& operator=(const TimedBlock&); // Not defined

    Timer& m_Timer;

  };


  /*! */
  inline TimedBlock::TimedBlock(Timer& Timer)
    :   m_Timer(Timer)
    {
      m_Timer.Start();
    }

    /*! */
    inline TimedBlock::~TimedBlock(void)
      {
	m_Timer.Stop();
      }
}

#endif
