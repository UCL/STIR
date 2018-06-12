/*
    Copyright (C) 2003-2011 Hammersmith Imanet Ltd
    Copyright (C) 2013-2014 University College London
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
  \ingroup listmode
  \brief Classes for listmode events for the ECAT 8 format
    
  \author Kris Thielemans
*/

#ifndef __stir_listmode_CListTimeECAT8_32bit_H__
#define __stir_listmode_CListTimeECAT8_32bit_H__

#include "stir/listmode/CListTime.h"
#include "stir/listmode/CListTimeDataECAT8_32bit.h"
#include "boost/static_assert.hpp"

START_NAMESPACE_STIR
namespace ecat {
    
//! A class for storing and using a timing 'event' from a listmode file from the ECAT 8_32bit scanner
/*! \ingroup listmode
 */
class CListTimeECAT8_32bit : public CListTime
{
 public:
  Succeeded init_from_data_ptr(const void * const ptr)
    {
      const char * const data_ptr = reinterpret_cast<const char * const >(ptr);
      std::copy(data_ptr, data_ptr+sizeof(this->raw), reinterpret_cast<char *>(&this->raw));
      return Succeeded::yes;
    }
  bool is_time() const
  { return this->data.type == 1U && this->data.deadtimeetc == 0U; }
  inline unsigned long get_time_in_millisecs() const
  { return static_cast<unsigned long>(this->data.time);  }
  inline Succeeded set_time_in_millisecs(const unsigned long time_in_millisecs)
  { 
    this->data.time = ((1U<<30)-1) & static_cast<unsigned>(time_in_millisecs); 
    // TODO return more useful value
    return Succeeded::yes;
  }

 private:
  BOOST_STATIC_ASSERT(sizeof(CListTimeDataECAT8_32bit)==4); 
  union 
  {
    CListTimeDataECAT8_32bit   data;
    boost::int32_t         raw;
  };
}; 


} // namespace ecat
END_NAMESPACE_STIR

#endif
