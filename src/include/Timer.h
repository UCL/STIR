// $Id$: $Date$

/*
  This file defines two classes: Timer and a derived class CPUTimer.
  Version 1.0 by Kris Thielemans


  Timer is a general base class for timers. Interface is like a stop watch:

  DerivedFromTimer t;
  t.start();
  // do things
  cerr << t.value();
  // do things
  t.stop();
  // do things
  t.restart()
  // etc

  Derived classes simply have to implement the virtual function 
  double get_current_value()


  CPUTimer  is derived from Timer, and hence has the same interface.
  CPUTimer::value() returns elapsed CPU time using the ANSI clock()
  function.  From the man page: "The reported time is the sum
  of the CPU time of the calling process and its terminated child
  processes for which it has executed wait, system, or pclose
  subroutines."

  Possible problem: 
  When the internal counter used by clock() wraps around, there is no way
  of detecting how many wraps occured, so I don't attempt to handle wraps 
  at all.
  For a SUN, the man page states this is every 2147 secs of CPU time (36 
  minutes !). AIX should be similar, as both have CLOCKS_PER_SEC = 10e6,
  and clock_t == int == 32 bit.
  Visual C++ is more sensible and has  CLOCKS_PER_SEC = 1000, so wrap around
  should occur only every 36000 minutes.
 
*/



class Timer
{
protected:
  bool running;
  double previous_value;
  double previous_total_value;
  
  virtual double get_current_value() const = 0;
  
public:
  Timer()
    {  
      //restart(); 
      running = false;
      previous_total_value = 0.;
      
    }
  ~Timer()
    {}
    
  void start() 
    { 
      if (!running)
      {
        running = true;
        previous_value = get_current_value();
      }
    }

  void stop()
    { 
      if (running)
        {
          previous_total_value += get_current_value() - previous_value;
          running = false;
        }
    }

  void restart() 
    { 
      previous_total_value = 0.;
      running = false;
      start();
    }

  double value() const
    { 
      double tmp = previous_total_value;  
      if (running)
         tmp += get_current_value() - previous_value;
      return tmp;
    }
};

#include <time.h>



class CPUTimer : public Timer
{
private:
  virtual double get_current_value() const
    {  
      return double( clock() ) / CLOCKS_PER_SEC;
    }
};

