// @(#)PTimer.h	1.6: 00/03/23
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

 \brief High-resolution wall-clock timer stir::HighResWallClockTimer

 \author Alexey Zverovich
 \author PARAPET project

  $Date$
  $Revision$

 Modification history:

  <TT>
  18 Aug 99 -- Mustapha Sadki -- corrected the formula for HighResWallClockTimer::Stop() & HighResWallClockTimer::GetTime() for WIN32\n
  08 Jul 99 -- Mustapha Sadki -- added global Ptimers and #define manipulations\n
  10 Apr 99 -- Alexey Zverovich -- Ported to Linux\n
  19 Oct 98 -- Alexey Zverovich -- Ported to Win32;\n
  19 Oct 98 -- Alexey Zverovich -- Slightly reorganised HighResWallClockTimer::Start() to reduce overhead\n
  16 Oct 98 -- Alexey Zverovich -- created
  </TT>

*/

#ifndef __stir_HIGHRESWALLCLOCKTIMER_H__
#define __stir_HIGHRESWALLCLOCKTIMER_H__

#if defined(_AIX) || defined(__sun) || defined(__linux__)
#   include <sys/time.h>
#elif defined(WIN32)
#   include <windows.h>
#endif

#include <assert.h>

#if defined(_MSC_VER) && (_MSC_VER < 1100) && !defined(bool)

enum bool
{
    false = 0,
    true  = (!false)
};

#define bool bool

#endif

namespace stir {

  /*! \brief High-resolution timer

  Typical usage:

  \code
  HighResWallClockTimer t;
  t.Start();
  do_something();
  t.Stop();
  cout << "do_something took " << t.GetTime() << " seconds" << endl;
  \endcode

  You have to call Reset() if you wish to use the same timer to measure
  several separate time intervals, otherwise the time will be accumulated.
  */

  class HighResWallClockTimer
  {
  public:

    //! Create a timer
    HighResWallClockTimer(void);
    //! Destroy a timer
    virtual ~HighResWallClockTimer(void);

    // Default copy ctor & assignment op are just fine

    //! Start a timer
    void Start(bool bReset = false);
    //! Stop a running timer
    void Stop(void);
    //! Reset a timer
    void Reset(void);
    //! Check if a timer is running
    bool IsRunning(void);

    //! Returns the number of whole seconds elapsed
    int GetSec(void);
    //! Returns the number of nanoseconds that elapsed on top of whatever GetSec() returned
    int GetNanosec(void);
    //! Returns the elapsed time (in seconds)
    double GetTime(void);

    //! Attempts to \e guess the timer resolution
    static int GetResolution(void); // in nanoseconds

  protected:
  private:

    bool  m_bRunning;
    int   m_Secs;
    int   m_Nanosecs;

#if defined(_AIX)
    timebasestruct_t m_Start;
    timebasestruct_t m_Finish;
#elif defined(__sun)
    hrtime_t         m_Start;
    hrtime_t         m_Finish;
#elif defined(WIN32)
    LARGE_INTEGER    m_Start;
    LARGE_INTEGER    m_Finish;
#elif defined(__linux__)
    timeval          m_Start;
    timeval          m_Finish;
#else
#error Do not know how to operate high-resolution timers on this OS
#endif

  };


  /*! The timer is not started, and elapsed time is set to zero. */
  inline HighResWallClockTimer::HighResWallClockTimer(void)
    :   m_bRunning(false)
    {
      Reset();
    };

    /*! The timer must not be running. */
    inline HighResWallClockTimer::~HighResWallClockTimer(void)
      {
	assert(!m_bRunning);
      };

    /*!
      \param bReset indicates whether the elapsed time should be reset before the timer is started.
      The timer must not be running.
    */
    inline void HighResWallClockTimer::Start(bool bReset /* = false */)
      {
	assert(!m_bRunning);
	if (bReset) Reset();
	m_bRunning = true;
#if defined(_AIX)
	read_real_time(&m_Start, TIMEBASE_SZ);
#elif defined(__sun)
	m_Start = gethrtime();
#elif defined(WIN32)
	BOOL Result = QueryPerformanceCounter(&m_Start);
	assert(Result); // if failed, high-resolution timers are not supported by this hardware
#elif defined(__linux__)
	// TODO: what will happen if the time gets changed?
	int Result = gettimeofday(&m_Start, NULL);
	assert(Result == 0);
#endif
      };

    /*!
      When a timer is stopped (=not running), GetSec(), GetNanosec(), and GetTime() can be
      used to obtain elapsed time.
    */
    inline void HighResWallClockTimer::Stop(void)
      {
	assert(m_bRunning);

#if defined(_AIX)

	read_real_time(&m_Finish, TIMEBASE_SZ);

	time_base_to_time(&m_Start, TIMEBASE_SZ);
	time_base_to_time(&m_Finish, TIMEBASE_SZ);

	int Secs = m_Finish.tb_high - m_Start.tb_high;
	int Nanosecs = m_Finish.tb_low - m_Start.tb_low;

	// If there was a carry from low-order to high-order during 
	// the measurement, we may have to undo it. 

	if (Nanosecs < 0)
	  {
	    Secs--;
	    Nanosecs += 1000000000;
	  };

	m_Secs += Secs;
	m_Nanosecs += Nanosecs;
	if (m_Nanosecs >= 1000000000)
	  {
	    m_Secs++;
	    m_Nanosecs -= 1000000000;
	  };

#elif defined(__sun)

	m_Finish = gethrtime();

	m_Secs += (m_Finish - m_Start) / 1000000000;
	m_Nanosecs += (m_Finish - m_Start) % 1000000000;
	if (m_Nanosecs >= 1000000000)
	  {
	    m_Secs++;
	    m_Nanosecs -= 1000000000;
	  };

#elif defined(WIN32)

	BOOL Result = QueryPerformanceCounter(&m_Finish);
	assert(Result); // if failed, high-resolution timers are not supported by this hardware

	LONGLONG Delta = m_Finish.QuadPart - m_Start.QuadPart;

	// MS 18-8-99  inserted this scope to correct the formula
	{
	  LARGE_INTEGER Freq;
	  BOOL          Result;
	  Result = QueryPerformanceFrequency(&Freq);         

	  assert(Result); // if failed, high-resolution timers are not supported by this hardware,//TODO: use GetTickCount()

	  m_Secs += static_cast<int>( Delta / Freq.QuadPart);
	  m_Nanosecs += static_cast<int>(Delta % Freq.QuadPart);

	  if (m_Nanosecs >= (int)Freq.QuadPart )
	    {
	      m_Secs++;
	      m_Nanosecs -= (int)Freq.QuadPart;
	    };
	  assert(m_Nanosecs >= 0 && m_Nanosecs < (int)Freq.QuadPart);

	}
	// and commented out 
	/*
	  int Resolution = GetResolution();

	  m_Secs += static_cast<int>(Delta / Resolution);
	  m_Nanosecs += static_cast<int>(Delta % Resolution);
	  if (m_Nanosecs >= 1000000000)
	  {
	  m_Secs++;
	  m_Nanosecs -= 1000000000;
	  };
	*/
#elif defined(__linux__)

	int Result = gettimeofday(&m_Finish, NULL);
	assert(Result == 0);

	m_Secs += (m_Finish.tv_sec - m_Start.tv_sec);
	int Microsecs = (m_Finish.tv_usec - m_Start.tv_usec);
	if (Microsecs < 0)
	  {
	    m_Secs--;
	    Microsecs += 1000000;
	  };

	m_Nanosecs += (Microsecs * 1000);

#endif

	assert(m_Secs >= 0);
	assert(m_Nanosecs >= 0 && m_Nanosecs < 1000000000);

	m_bRunning = false;
      };

    /*! Sets the elapsed time to zero. The timer must not be running. */
    inline void HighResWallClockTimer::Reset(void)
      {
	assert(!m_bRunning);
	m_Secs = m_Nanosecs = 0;
      };

    /*! Returns true if the timer is running, or false otherwise */
    inline bool HighResWallClockTimer::IsRunning(void)
      {
	return m_bRunning;
      };

    /*! The timer must not be running */
    inline int HighResWallClockTimer::GetSec(void)
      {
	assert(!m_bRunning);
	return m_Secs;
      };

    /*! The timer must not be running */
    inline int HighResWallClockTimer::GetNanosec(void)
      {
	assert(!m_bRunning);
	return m_Nanosecs;
      };

    /*! The timer must not be running */
    inline double HighResWallClockTimer::GetTime(void)
      {
	// MS 18-8-99 added 
#ifdef WIN32
	LARGE_INTEGER Freq;
	BOOL          Result;
	Result = QueryPerformanceFrequency(&Freq);         
	return GetSec() + double(GetNanosec()) / (double)Freq.QuadPart;

#else
	return GetSec() + double(GetNanosec()) / 1e9;
#endif
      };

    // HighResWallClockTimer.cxx -- high-resolution timers implementation

    // Modification history:
    //
    //  19 Oct 98 -- Alexey Zverovich -- created

    /*!
      GetResolution() is by no means accurate.
      The only thing that is guranteed is that timer resolution
      is not less than the value returned by GetResolution().

      On Win32, returns more or less precise resolution (can be
      slightly rounded if the actual resolution can't be expressed
      as an integer number of nanoseconds)

      \return Estimated timer resolution (in nanoseconds)
    */

    inline int HighResWallClockTimer::GetResolution(void)
      {

	/*static*/ int Resolution = -1; // cached resolution

	if (Resolution > 0) return Resolution;

#if defined(_AIX)

	timebasestruct_t Start;
	timebasestruct_t Current;

	read_real_time(&Start, TIMEBASE_SZ);
	time_base_to_time(&Start, TIMEBASE_SZ);

	int MinSecs        = -1; // no minimum yet
	int MinNanosecs    = -1;
	int SecondsElapsed = 0;

	do
	  {
	    timebasestruct_t Times[2];

	    read_real_time(&Times[0], TIMEBASE_SZ);
	    do
	      {
		read_real_time(&Times[1], TIMEBASE_SZ);
		assert(Times[1].flag == Times[0].flag);
	      }
	    while (Times[1].tb_high == Times[0].tb_high &&
		   Times[1].tb_low  == Times[0].tb_low);

	    time_base_to_time(&Times[0], TIMEBASE_SZ);
	    time_base_to_time(&Times[1], TIMEBASE_SZ);

	    int Secs = Times[1].tb_high - Times[0].tb_high;
	    int Nanosecs = Times[1].tb_low - Times[0].tb_low;

	    // If there was a carry from low-order to high-order during 
	    // the measurement, we may have to undo it. 

	    if (Nanosecs < 0)
	      {
		Secs--;
		Nanosecs += 1000000000;
	      };

	    assert(Secs >= 0 && Nanosecs >= 0 && Nanosecs < 1000000000);
	    assert(Secs != 0 || Nanosecs != 0); // shouldn't be the same as we checked in the 'do' loop

	    //        cout << "[" << Secs << " " << Nanosecs << "]" << endl;

	    if (MinSecs < 0 || Secs < MinSecs || (Secs == MinSecs && Nanosecs < MinNanosecs))
	      {
		MinSecs = Secs;
		MinNanosecs = Nanosecs;
	      };

	    read_real_time(&Current, TIMEBASE_SZ);
	    time_base_to_time(&Current, TIMEBASE_SZ);

	    SecondsElapsed = Current.tb_high - Start.tb_high;
	    if (Current.tb_low < Start.tb_low) SecondsElapsed--;
	  }
	while (SecondsElapsed < 1);

	assert(MinSecs < 2); // otherwise it's too coarse for this measurement

	Resolution = MinSecs * 1000000000 + MinNanosecs;

#elif defined(__sun)

	hrtime_t Start       = gethrtime();
	hrtime_t Current;

	hrtime_t MinNanosecs = -1;

	do
	  {
	    hrtime_t Times[2];

	    Times[0] = gethrtime();

	    while ((Times[1] = gethrtime()) == Times[0]) /* do nothing */;

	    hrtime_t Nanosecs = Times[1] - Times[0];

	    assert(Nanosecs > 0);

	    if (MinNanosecs < 0 || Nanosecs < MinNanosecs)
	      {
		MinNanosecs = Nanosecs;
	      };

	    Current = gethrtime();
	  }
	while (Current - Start < 1000000000); // 1 second

	assert(MinNanosecs < 2000000000); // otherwise it's too coarse for this measurement

	Resolution = static_cast<int>(MinNanosecs);

#elif defined(WIN32)

	LARGE_INTEGER Freq;
	BOOL          Result;

	Result = QueryPerformanceFrequency(&Freq);
	assert(Result); // if failed, high-resolution timers are not supported by this hardware

	Resolution = static_cast<int>(1000000000 / Freq.QuadPart);

#elif defined(__linux__)

	//TODO

#endif

	assert(Resolution > 0);

	return Resolution;

      };

}


#endif // __HIGHRESWALLCLOCKTIMER_H__

