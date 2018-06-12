//
//
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
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
  \brief Classes for listmode events for the ECAT 966 (aka Exact 3d)
    
  \author Nikos Efthimiou  
  \author Kris Thielemans
      
*/
#ifndef __stir_listmode_CListTimeECAT966_H__
#define __stir_listmode_CCListTimeECAT966_H__

#include "stir/listmode/CListTime.h"
#include "stir/listmode/CListGatingInput.h"
#include "stir/Succeeded.h"
#include "boost/static_assert.hpp"

START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7

//! A class for storing and using a timing 'event' from a listmode file from the ECAT 966 scanner
/*! \ingroup listmode
 */
class CListTimeECAT966 : public CListTime, public CListGatingInput
{
 public:
  Succeeded init_from_data_ptr(const void * const ptr)
    {
      const char * const data_ptr = reinterpret_cast<const char * const >(ptr);
      std::copy(data_ptr, data_ptr+sizeof(this->raw), reinterpret_cast<char *>(&this->raw));
      return Succeeded::yes;
    }
  bool is_time() const
  { return this->data.type == 1U; }
  inline unsigned long get_time_in_millisecs() const
  { return static_cast<unsigned long>(this->data.time);  }
  inline Succeeded set_time_in_millisecs(const unsigned long time_in_millisecs)
  { 
    this->data.time = ((1U<<28)-1) & static_cast<unsigned>(time_in_millisecs); 
    // TODO return more useful value
    return Succeeded::yes;
  }
  inline unsigned int get_gating() const
  { return this->data.gating; }
  inline Succeeded set_gating(unsigned int g)
  { this->data.gating = g & 0xf; return this->data.gating==g ? Succeeded::yes : Succeeded::no;}

 private:
  BOOST_STATIC_ASSERT(sizeof(CListTimeDataECAT966)==4); 
  union 
  {
    CListTimeDataECAT966   data;
    boost::int32_t         raw;
  };
};

END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR

#endif
