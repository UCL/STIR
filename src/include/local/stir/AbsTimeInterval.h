//
// $Id$
//
/*
    Copyright (C) 2005- $Date$ , Hammersmith Imanet Ltd
    For internal GE use only
*/
#ifndef __stir_AbsTimeInterval__H__
#define __stir_AbsTimeInterval__H__
/*!
  \file
  \ingroup buildblock

  \brief Declaration of class stir::AbsTimeInterval

  \author Kris Thielemans
  $Date$
  $Revision$
*/

#include "stir/RegisteredObject.h"
#include "stir/ParsingObject.h"
#include <ctime>

# ifdef BOOST_NO_STDC_NAMESPACE
namespace std { using ::time_t; }
#endif


START_NAMESPACE_STIR

/*! \ingroup buildblock

  \brief Base class for specifying a time interval (in absolute time)

  Absolute time means at present 'secs since midnight 1/1/1970 UTC'

  \todo Probably should allow for milli (or micro)secs.
*/
class AbsTimeInterval: public RegisteredObject<AbsTimeInterval>,
                           public ParsingObject
{

public:
  virtual ~AbsTimeInterval() {}
  AbsTimeInterval()
    :
    _start_time_in_secs_since_1970(0),
    _end_time_in_secs_since_1970(0)
    {}
  AbsTimeInterval( std::time_t start_time_in_secs_since_1970,
		   std::time_t end_time_in_secs_since_1970)
    : 
    _start_time_in_secs_since_1970(start_time_in_secs_since_1970),
    _end_time_in_secs_since_1970(end_time_in_secs_since_1970)
    {}  		   


  std::time_t  get_start_time_in_secs_since_1970() const
    { return _start_time_in_secs_since_1970; }
  std::time_t get_end_time_in_secs_since_1970() const
    { return _end_time_in_secs_since_1970; }
  std::time_t get_duration_in_secs() const
    { return _end_time_in_secs_since_1970 - _start_time_in_secs_since_1970; }
  
 protected:
  std::time_t _start_time_in_secs_since_1970;
  std::time_t _end_time_in_secs_since_1970;

};

END_NAMESPACE_STIR

#endif
